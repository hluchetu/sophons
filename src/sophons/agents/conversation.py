from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from sophons.models.messages import Message

if TYPE_CHECKING:
    from sophons.models.chat import ChatModel


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TokenCounter(Protocol):
    """Counts tokens for a single message."""

    def count_message(self, message: Message) -> int:
        ...


class ConversationManager(Protocol):
    """
    Decides which messages from the full history are passed to the model.

    ``prepare`` receives the complete message list and returns the slice
    the model should see.  It must not mutate the input list.
    """

    def prepare(
        self,
        messages: list[Message],
        current_input: str | None = None,
    ) -> list[Message]:
        ...


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
    """Passed to every manager's ``prepare`` call."""

    current_input: str | None = None


# ---------------------------------------------------------------------------
# SlidingWindowManager
# ---------------------------------------------------------------------------


class SlidingWindowManager:
    """
    Keeps the last ``max_messages`` messages.

    System messages at the front are always preserved regardless of the
    window size.
    """

    def __init__(
        self,
        max_messages: int,
        preserve_system_messages: bool = True,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0.")
        self._max_messages = max_messages
        self._preserve_system = preserve_system_messages

    def prepare(
        self,
        messages: list[Message],
        current_input: str | None = None,
    ) -> list[Message]:
        if not messages:
            return messages

        system_messages: list[Message] = []
        rest: list[Message] = []

        if self._preserve_system:
            for message in messages:
                if message.role == "system":
                    system_messages.append(message)
                else:
                    rest.append(message)
        else:
            rest = list(messages)

        trimmed = rest[-self._max_messages :]
        return [*system_messages, *trimmed]


# ---------------------------------------------------------------------------
# TokenBudgetManager
# ---------------------------------------------------------------------------


class ContextBudgetExceededError(Exception):
    """Raised when preserved messages alone exceed the token budget."""


class TokenBudgetManager:
    """
    Keeps as many messages as fit within a token budget, always keeping the
    most recent messages and any pinned or system messages.

    Tool call + tool result groups are treated as a single unit so a tool
    interaction is never split across the context boundary.
    """

    def __init__(
        self,
        max_tokens: int,
        token_counter: TokenCounter,
        preserve_system_messages: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        self._max_tokens = max_tokens
        self._token_counter = token_counter
        self._preserve_system = preserve_system_messages

    # ------------------------------------------------------------------
    # ConversationManager
    # ------------------------------------------------------------------

    def prepare(
        self,
        messages: list[Message],
        current_input: str | None = None,
    ) -> list[Message]:
        if not messages:
            return messages

        preserved, candidates = self._split_preserved(messages)
        used = self._count(preserved)

        if used > self._max_tokens:
            raise ContextBudgetExceededError(
                "Preserved messages already exceed the token budget."
            )

        selected_units: list[list[Message]] = []
        units = self._group_units(candidates)

        for unit in reversed(units):
            cost = self._count(unit)
            if used + cost > self._max_tokens:
                if not selected_units:
                    raise ContextBudgetExceededError(
                        "The newest message group exceeds the remaining token budget."
                    )
                break
            selected_units.append(unit)
            used += cost

        selected: list[Message] = [m for unit in reversed(selected_units) for m in unit]
        kept_ids = {m.id for m in [*preserved, *selected] if m.id is not None}

        # Preserve ordering from original list
        return [m for m in messages if m.id in kept_ids or m.id is None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_preserved(
        self,
        messages: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        preserved_ids: set[str] = set()
        preserved: list[Message] = []
        candidate_start = 0

        if self._preserve_system:
            for message in messages:
                if message.role != "system":
                    break
                if message.id is not None:
                    preserved_ids.add(message.id)
                preserved.append(message)
                candidate_start += 1

        for message in messages:
            if _is_pinned(message) and message.id not in preserved_ids:
                if message.id is not None:
                    preserved_ids.add(message.id)
                preserved.append(message)

        candidates = [
            m for m in messages[candidate_start:]
            if m.id not in preserved_ids
        ]
        return preserved, candidates

    def _group_units(self, messages: list[Message]) -> list[list[Message]]:
        """Group assistant+tool call chains into atomic units."""
        units: list[list[Message]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            unit = [message]
            index += 1
            if message.role == "assistant" and _has_tool_calls(message):
                while index < len(messages) and messages[index].role == "tool":
                    unit.append(messages[index])
                    index += 1
            units.append(unit)
        return units

    def _count(self, messages: list[Message]) -> int:
        return sum(self._token_counter.count_message(m) for m in messages)


# ---------------------------------------------------------------------------
# ToolInteractionCompactor
# ---------------------------------------------------------------------------


class ToolInteractionCompactor:
    """
    Replaces old tool-call + tool-result groups with a compact summary
    message, keeping only the most recent ``keep_recent`` interactions
    in full.

    This reduces context size without losing the record of what the agent
    did.
    """

    def __init__(
        self,
        keep_recent: int,
        max_result_chars: int = 500,
    ) -> None:
        if keep_recent < 0:
            raise ValueError("keep_recent must be >= 0.")
        if max_result_chars <= 0:
            raise ValueError("max_result_chars must be > 0.")
        self._keep_recent = keep_recent
        self._max_result_chars = max_result_chars

    def prepare(
        self,
        messages: list[Message],
        current_input: str | None = None,
    ) -> list[Message]:
        units = self._group_units(messages)
        tool_unit_indices = [
            i for i, unit in enumerate(units) if self._is_tool_interaction(unit)
        ]
        compact_count = max(0, len(tool_unit_indices) - self._keep_recent)
        compact_indices = set(tool_unit_indices[:compact_count])

        result: list[Message] = []
        for i, unit in enumerate(units):
            if i in compact_indices and not any(_is_pinned(m) for m in unit):
                result.append(self._compact(unit))
            else:
                result.extend(unit)
        return result

    def _group_units(self, messages: list[Message]) -> list[list[Message]]:
        units: list[list[Message]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            unit = [message]
            index += 1
            if message.role == "assistant" and _has_tool_calls(message):
                while index < len(messages) and messages[index].role == "tool":
                    unit.append(messages[index])
                    index += 1
            units.append(unit)
        return units

    def _is_tool_interaction(self, unit: list[Message]) -> bool:
        return bool(unit) and unit[0].role == "assistant" and _has_tool_calls(unit[0])

    def _compact(self, unit: list[Message]) -> Message:
        assistant = unit[0]
        tool_results = unit[1:]
        covered_ids = [m.id for m in unit if m.id is not None]

        lines = [
            "Compacted older tool interaction.",
            "The raw tool call and tool result messages were omitted from model context.",
            "",
            "Tool calls:",
        ]
        for tc in _tool_calls(assistant):
            name = tc.get("name", "unknown")
            args = tc.get("arguments", tc.get("input", {}))
            lines.append(f"- {name}: {args}")

        if assistant.content.strip():
            lines += ["", f"Assistant note: {assistant.content.strip()}"]

        if tool_results:
            lines += ["", "Tool results:"]
            for m in tool_results:
                tool_name = str(m.metadata.get("name", "tool"))
                lines.append(f"- {tool_name}: {self._truncate(m.content)}")

        return Message(
            role="system",
            content="\n".join(lines),
            metadata={
                "kind": "tool_interaction_compaction",
                "covered_item_ids": covered_ids,
            },
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_result_chars:
            return text
        return f"{text[: self._max_result_chars].rstrip()}..."


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
    ) -> None:
        if trigger_message_count is None and trigger_token_count is None:
            raise ValueError(
                "Provide at least one of trigger_message_count or trigger_token_count."
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

    def prepare(
        self,
        messages: list[Message],
        current_input: str | None = None,
    ) -> list[Message]:
        if not self._should_summarize(messages):
            return messages

        old = messages[: -self._keep_recent]
        recent = messages[-self._keep_recent :]

        prev_summary_index = self._latest_summary_index(old)
        prev_summary = old[prev_summary_index] if prev_summary_index is not None else None
        candidates = (
            old[prev_summary_index + 1 :] if prev_summary_index is not None else old
        )
        pinned_old = [m for m in old if _is_pinned(m) and m is not prev_summary]
        summarizable = [
            m for m in candidates
            if not _is_pinned(m) and not self._is_summary(m)
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

    def _should_summarize(self, messages: list[Message]) -> bool:
        if len(messages) <= self._keep_recent:
            return False
        if self._trigger_token_count is not None and self._token_counter is not None:
            total = sum(self._token_counter.count_message(m) for m in messages)
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
            response = self._model.chat(prompt_messages)
        except Exception as exc:
            raise SummarizationError(
                "Failed to summarize conversation messages."
            ) from exc
        return response.content.strip()


# ---------------------------------------------------------------------------
# ManagerPipeline
# ---------------------------------------------------------------------------


class ManagerPipeline:
    """
    Chains multiple ``ConversationManager`` instances into one.

    Each manager's output becomes the next manager's input.  Useful for
    combining, for example, ``ToolInteractionCompactor`` followed by
    ``TokenBudgetManager``.
    """

    def __init__(self, managers: list[ConversationManager]) -> None:
        if not managers:
            raise ValueError("ManagerPipeline requires at least one manager.")
        self._managers = managers

    def prepare(
        self,
        messages: list[Message],
        current_input: str | None = None,
    ) -> list[Message]:
        result = messages
        for manager in self._managers:
            result = manager.prepare(result, current_input)
        return result
