from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from sophons.models.messages import Message

if TYPE_CHECKING:
    from sophons.models.chat import ChatModel


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TokenCounter(Protocol):
    """Counts tokens for a single message."""

    def count_message(self, message: Message) -> int: ...


class ApproximateTokenCounter:
    """
    Estimates tokens by character count, without a tokenizer.

    Exact counting needs the model's own tokenizer, which means a provider
    dependency or a network round trip. For deciding how much history fits
    in a budget, an estimate is enough — budgets are set below the real
    limit anyway, so the cost of being slightly wrong is a few wasted
    tokens rather than a rejected request.

    It is deliberately named "approximate" so nobody mistakes it for exact.
    Under-counting is the dangerous direction, so the defaults lean high:
    tool calls in metadata are counted (they are sent to the model and are
    easy to forget), and a per-message overhead covers the role and
    delimiters every provider adds.

    Args:
        chars_per_token:      Average characters per token. 4.0 is the usual
                              rule of thumb for English prose; code and
                              non-Latin scripts pack fewer characters per
                              token, so lower it if you work in those.
        per_message_overhead: Tokens charged per message for role and
                              formatting, independent of content.
    """

    def __init__(
        self,
        chars_per_token: float = 4.0,
        per_message_overhead: int = 4,
    ) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be greater than 0.")
        if per_message_overhead < 0:
            raise ValueError("per_message_overhead must not be negative.")
        self._chars_per_token = chars_per_token
        self._per_message_overhead = per_message_overhead

    def count_message(self, message: Message) -> int:
        characters = len(message.content)

        # Tool calls travel to the model as serialized JSON, so a message
        # with empty content is not free.
        for call in _tool_calls(message):
            characters += len(str(call.get("name", "")))
            characters += len(str(call.get("input", call.get("arguments", ""))))

        estimate = characters / self._chars_per_token
        return self._per_message_overhead + math.ceil(estimate)


class ConversationManager(Protocol):
    """
    Decides which messages from the full history are passed to the model.

    Two moments, two methods:

    - ``prepare`` runs before every model call and returns the slice the
      model should see. It must not mutate the input list.
    - ``reduce_context`` runs after the model has *rejected* a request for
      being too large, and returns something smaller to retry with. An
      estimate can be wrong; this is the path that does not depend on
      guessing correctly.
    """

    def prepare(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
    ) -> list[Message]: ...

    def reduce_context(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
        error: Exception | None = None,
    ) -> list[Message]: ...


class NullConversationManager:
    """
    Passes the entire history through untouched.

    This is what an agent does when no manager is configured, but saying it
    out loud is worth something: unbounded history is a decision with a
    failure mode, not a neutral default. Every turn is replayed on every
    later call until the request exceeds the model's context window.

    Use it when conversations are known to be short, or as the base case in
    a benchmark against a real strategy.
    """

    def prepare(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
    ) -> list[Message]:
        return messages

    def reduce_context(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
        error: Exception | None = None,
    ) -> list[Message]:
        # Nothing to give back. Re-raising is honest: a manager that declines
        # to manage cannot rescue an overflow, and pretending otherwise would
        # send the identical request again.
        if error is not None:
            raise error
        return messages


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
# Sophons Message is a plain frozen dataclass.  Extended fields that some
# managers need (pinned, tool_calls) are stored in message.metadata so the
# core type stays lightweight.


def _is_pinned(message: Message) -> bool:
    return bool(message.metadata.get("pinned", False))


def _has_tool_calls(message: Message) -> bool:
    return bool(message.metadata.get("tool_calls"))


def _tool_calls(message: Message) -> list[dict[str, Any]]:
    return list(message.metadata.get("tool_calls") or [])


# ---------------------------------------------------------------------------
# Processing context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepareContext:
    """
    What a manager knows about the run when deciding what to keep.

    Managers stay pure functions of ``(messages, context)``. Anything a
    strategy needs beyond the messages themselves arrives here rather than
    being captured at construction, which is what lets a budget be expressed
    as a fraction of the model's real window instead of a number the caller
    has to guess.

    Attributes:
        current_input:  The user message this run is answering, when known.
        token_counter:  Counter for sizing messages. Managers given their own
                        counter should prefer that; this is the fallback.
        context_window: The model's total context size in tokens, when the
                        model declares one. ``None`` means unknown, and any
                        ratio-based behaviour must degrade rather than guess.
    """

    current_input: str | None = None
    token_counter: TokenCounter | None = None
    context_window: int | None = None

    def budget(self, ratio: float) -> int | None:
        """
        Tokens available at ``ratio`` of the model's window.

        Returns ``None`` when the window is unknown, so callers can fall back
        to an absolute setting instead of compressing against a number that
        does not exist.
        """
        if self.context_window is None:
            return None
        return int(self.context_window * ratio)

# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------
# These are invariants, not strategies. Every manager needs them, and none of
# them is a decision a caller should have to make.


def _group_units(
    messages: list[Message],
    indices: list[int] | None = None,
) -> list[list[int]]:
    """
    Group message indices into atomic units.

    An assistant message carrying tool calls belongs with the tool results
    that answer it: sending a tool result whose originating call was dropped
    produces a malformed request that some providers reject outright. Units
    exist so no strategy can split that pair, whatever else it discards.
    """
    positions = list(range(len(messages))) if indices is None else list(indices)
    units: list[list[int]] = []
    cursor = 0
    while cursor < len(positions):
        index = positions[cursor]
        unit = [index]
        cursor += 1
        if messages[index].role == "assistant" and _has_tool_calls(messages[index]):
            while (
                cursor < len(positions)
                and messages[positions[cursor]].role == "tool"
            ):
                unit.append(positions[cursor])
                cursor += 1
        units.append(unit)
    return units


def _truncate_tool_results(
    messages: list[Message],
    max_chars: int,
) -> list[Message]:
    """
    Shrink oversized tool results, keeping both ends.

    Run before anything is dropped. A 10 KB tool result costs a lot of
    context and little meaning; discarding the message that holds it costs
    the agent its record of what it did. Head and tail are kept because the
    useful parts of a long result usually sit at the edges — what was asked
    for, and how it ended.
    """
    truncated: list[Message] = []
    for message in messages:
        if message.role == "tool" and len(message.content) > max_chars:
            head = max_chars // 2
            tail = max_chars - head
            body = (
                f"{message.content[:head].rstrip()}"
                f"\n...[{len(message.content) - max_chars} characters truncated]...\n"
                f"{message.content[-tail:].lstrip()}"
            )
            truncated.append(replace(message, content=body))
        else:
            truncated.append(message)
    return truncated


# ---------------------------------------------------------------------------
# SlidingWindowManager
# ---------------------------------------------------------------------------


class ContextBudgetExceededError(Exception):
    """Raised when preserved messages alone exceed the token budget."""


class SlidingWindowManager:
    """
    Keeps the most recent history and drops the rest.

    The window is measured either in messages or in tokens — the same
    strategy, sized differently, so it is one argument rather than two
    classes. Pass exactly one of ``max_messages`` or ``max_tokens``.

    Two behaviours are invariants rather than options:

    - A tool call and its results are kept or dropped together, never split.
    - Oversized tool results are truncated before any message is discarded,
      because shrinking a result loses less than losing the record of it.

    System messages and pinned messages are preserved regardless of the
    window, and the original ordering of what survives is left alone.

    Args:
        max_messages:            Window size in messages.
        max_tokens:              Window size in tokens. Mutually exclusive
                                 with ``max_messages``.
        token_counter:           Used when sizing in tokens. Defaults to
                                 ``ApproximateTokenCounter``.
        truncate_tool_results:   Shrink long tool results before dropping.
        max_result_chars:        Size a tool result is truncated to.
        preserve_system_messages: Keep system messages outside the window.
    """

    def __init__(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        token_counter: TokenCounter | None = None,
        truncate_tool_results: bool = True,
        max_result_chars: int = 500,
        preserve_system_messages: bool = True,
    ) -> None:
        if (max_messages is None) == (max_tokens is None):
            raise ValueError("Provide exactly one of max_messages or max_tokens.")
        if max_messages is not None and max_messages <= 0:
            raise ValueError("max_messages must be greater than 0.")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        if max_result_chars <= 0:
            raise ValueError("max_result_chars must be greater than 0.")

        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._token_counter = token_counter
        self._truncate_tool_results = truncate_tool_results
        self._max_result_chars = max_result_chars
        self._preserve_system = preserve_system_messages

    def prepare(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
    ) -> list[Message]:
        if not messages:
            return messages

        working = messages
        if self._truncate_tool_results:
            working = _truncate_tool_results(working, self._max_result_chars)

        # Selection works on positions rather than message ids: an earlier
        # version filtered by id set, so messages without an id escaped the
        # budget entirely — and model adapters return assistant messages with
        # no id.
        preserved = {
            index
            for index, message in enumerate(working)
            if (self._preserve_system and message.role == "system")
            or _is_pinned(message)
        }
        candidates = [i for i in range(len(working)) if i not in preserved]
        units = _group_units(working, candidates)

        if self._max_tokens is not None:
            selected = self._select_by_tokens(working, units, preserved, context)
        else:
            selected = self._select_by_count(units)

        keep = preserved | selected
        return [message for index, message in enumerate(working) if index in keep]

    def reduce_context(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
        error: Exception | None = None,
    ) -> list[Message]:
        """Halve the window after the model rejected the request as too large."""
        halved = SlidingWindowManager(
            max_messages=(
                max(1, self._max_messages // 2)
                if self._max_messages is not None
                else None
            ),
            max_tokens=(
                max(1, self._max_tokens // 2) if self._max_tokens is not None else None
            ),
            token_counter=self._token_counter,
            truncate_tool_results=self._truncate_tool_results,
            max_result_chars=self._max_result_chars,
            preserve_system_messages=self._preserve_system,
        )
        return halved.prepare(messages, context)

    # ------------------------------------------------------------------

    def _counter(self, context: PrepareContext | None) -> TokenCounter:
        return (
            self._token_counter
            or (context.token_counter if context else None)
            or ApproximateTokenCounter()
        )

    def _select_by_tokens(
        self,
        messages: list[Message],
        units: list[list[int]],
        preserved: set[int],
        context: PrepareContext | None,
    ) -> set[int]:
        assert self._max_tokens is not None
        counter = self._counter(context)
        used = sum(counter.count_message(messages[i]) for i in preserved)
        if used > self._max_tokens:
            raise ContextBudgetExceededError(
                "Preserved messages already exceed the token budget."
            )

        selected: set[int] = set()
        for unit in reversed(units):
            cost = sum(counter.count_message(messages[i]) for i in unit)
            if used + cost > self._max_tokens:
                if not selected:
                    raise ContextBudgetExceededError(
                        "The newest message group exceeds the remaining token budget."
                    )
                break
            selected.update(unit)
            used += cost
        return selected

    def _select_by_count(self, units: list[list[int]]) -> set[int]:
        assert self._max_messages is not None
        selected: set[int] = set()
        kept = 0
        for unit in reversed(units):
            # The newest unit is always kept whole, even if it alone exceeds
            # the window — a truncated tool interaction is worse than a wide
            # one.
            if kept and kept + len(unit) > self._max_messages:
                break
            selected.update(unit)
            kept += len(unit)
        return selected


# ---------------------------------------------------------------------------
# SummarizingManager
# ---------------------------------------------------------------------------


class SummarizationError(Exception):
    """Raised when the model fails to produce a summary."""


_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Produce a concise, factual summary of the conversation below. "
    "Preserve all decisions, facts, and action outcomes. "
    "Use plain prose. Do not add commentary."
)

_SUMMARY_USER_TEMPLATE = "Summarize the following conversation:\n\n{conversation}"


class SummarizingManager:
    """
    Summarizes old messages with an LLM once a trigger threshold is crossed,
    keeping only the most recent ``keep_recent_messages`` in full.

    Trigger can be based on message count, token count, or both.
    Consecutive summaries are merged so only one summary message appears in
    context at a time.
    """

    def __init__(
        self,
        model: ChatModel,
        keep_recent_messages: int,
        trigger_message_count: int | None = None,
        trigger_token_count: int | None = None,
        token_counter: TokenCounter | None = None,
        compression_threshold: float | None = None,
    ) -> None:
        if (
            trigger_message_count is None
            and trigger_token_count is None
            and compression_threshold is None
        ):
            raise ValueError(
                "Provide at least one of trigger_message_count, "
                "trigger_token_count, or compression_threshold."
            )
        if compression_threshold is not None and not 0 < compression_threshold <= 1:
            raise ValueError(
                "compression_threshold must be between 0 and 1 exclusive of 0."
            )
        if trigger_token_count is not None and token_counter is None:
            raise ValueError(
                "token_counter is required when trigger_token_count is set."
            )
        if (
            trigger_message_count is not None
            and keep_recent_messages >= trigger_message_count
        ):
            raise ValueError(
                "keep_recent_messages must be smaller than trigger_message_count."
            )
        if keep_recent_messages <= 0:
            raise ValueError("keep_recent_messages must be > 0.")

        self._model = model
        self._keep_recent = keep_recent_messages
        self._trigger_message_count = trigger_message_count
        self._trigger_token_count = trigger_token_count
        self._token_counter = token_counter
        self._compression_threshold = compression_threshold

    def prepare(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
    ) -> list[Message]:
        if not self._should_summarize(messages, context):
            return messages

        old = messages[: -self._keep_recent]
        recent = messages[-self._keep_recent :]

        prev_summary_index = self._latest_summary_index(old)
        prev_summary = (
            old[prev_summary_index] if prev_summary_index is not None else None
        )
        candidates = (
            old[prev_summary_index + 1 :] if prev_summary_index is not None else old
        )
        pinned_old = [m for m in old if _is_pinned(m) and m is not prev_summary]
        summarizable = [
            m for m in candidates if not _is_pinned(m) and not self._is_summary(m)
        ]

        if not summarizable:
            carried = [prev_summary] if prev_summary is not None else []
            return [*pinned_old, *carried, *recent]

        new_summary_text = self._call_model(summarizable)
        summary_msg = Message(
            role="system",
            content=self._build_content(prev_summary, new_summary_text),
            metadata={
                "kind": "conversation_summary",
                "covered_item_ids": self._covered_ids(prev_summary, summarizable),
            },
        )
        return [*pinned_old, summary_msg, *recent]

    # ------------------------------------------------------------------

    def _should_summarize(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
    ) -> bool:
        if len(messages) <= self._keep_recent:
            return False

        counter = self._token_counter or (context.token_counter if context else None)

        # Proactive: compress at a fraction of the model's real window, so the
        # threshold does not have to be guessed per model. Only usable when the
        # model declares a window and a counter is available; otherwise fall
        # through to the absolute triggers below.
        if self._compression_threshold is not None and counter is not None:
            budget = context.budget(self._compression_threshold) if context else None
            if budget is not None:
                total = sum(counter.count_message(m) for m in messages)
                return total > budget

        if self._trigger_token_count is not None and counter is not None:
            total = sum(counter.count_message(m) for m in messages)
            return total > self._trigger_token_count
        if self._trigger_message_count is not None:
            return len(messages) >= self._trigger_message_count
        return False

    def _latest_summary_index(self, messages: list[Message]) -> int | None:
        for i in range(len(messages) - 1, -1, -1):
            if self._is_summary(messages[i]):
                return i
        return None

    def _is_summary(self, message: Message) -> bool:
        return message.metadata.get("kind") == "conversation_summary"

    def _build_content(self, previous: Message | None, new_text: str) -> str:
        parts: list[str] = []
        if previous is not None:
            prefix = "Conversation summary so far:\n"
            body = previous.content
            parts.append(
                body[len(prefix) :].strip() if body.startswith(prefix) else body.strip()
            )
        parts.append(new_text)
        return "Conversation summary so far:\n" + "\n".join(parts)

    def _covered_ids(
        self,
        previous: Message | None,
        summarized: list[Message],
    ) -> list[str]:
        ids: list[str] = []
        if previous is not None:
            raw = previous.metadata.get("covered_item_ids", [])
            if isinstance(raw, list):
                ids.extend(str(i) for i in raw)
            elif previous.id is not None:
                ids.append(previous.id)
        ids.extend(m.id for m in summarized if m.id is not None)
        return ids

    def reduce_context(
        self,
        messages: list[Message],
        context: PrepareContext | None = None,
        error: Exception | None = None,
    ) -> list[Message]:
        """
        Summarize regardless of the trigger, keeping fewer messages verbatim.

        The trigger already failed to fire early enough — the model rejected
        the request — so waiting for it again is not an option.
        """
        keep = max(1, self._keep_recent // 2)
        forced = SummarizingManager(
            model=self._model,
            keep_recent_messages=keep,
            # Just above what is kept, so anything older is summarized on this
            # call rather than waiting for a threshold that has already proven
            # too slow.
            trigger_message_count=keep + 1,
            token_counter=self._token_counter,
        )
        return forced.prepare(messages, context)

    def _call_model(self, messages: list[Message]) -> str:
        conversation_text = "\n".join(f"{m.role}: {m.content}" for m in messages)
        prompt_messages = [
            Message(role="system", content=_SUMMARY_SYSTEM_PROMPT),
            Message(
                role="user",
                content=_SUMMARY_USER_TEMPLATE.format(conversation=conversation_text),
            ),
        ]
        try:
            response = self._model.invoke(prompt_messages)
        except AttributeError:
            # A model that cannot be called at all is a wiring bug, not a
            # summarization failure — let it surface instead of being
            # relabelled as one.
            raise
        except Exception as exc:
            raise SummarizationError(
                "Failed to summarize conversation messages."
            ) from exc
        return response.content.strip()
