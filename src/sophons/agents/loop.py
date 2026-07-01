from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from sophons.agents.conversation import ConversationManager
from sophons.agents.hooks import (
    AfterModelCall,
    AfterToolCall,
    AgentFailed,
    AgentFinished,
    AgentStarted,
    BeforeModelCall,
    BeforeToolCall,
    HookRegistry,
    MessageAdded,
)
from sophons.agents.responses import (
    AgentMetrics,
    AgentResult,
    StopReason,
    ToolResult,
    ToolUse,
)
from sophons.agents.retry import RetryStrategy, no_retry
from sophons.agents.state import RunLimits, RunState
from sophons.models.messages import Message
from sophons.tools.base import AsyncTool, Tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """
    Drives a single agent run from first user message to final answer.

    The loop is intentionally thin:

    - It prepares context, calls the model, executes tools, and tracks state.
    - All side effects (logging, tracing, session saving) happen via hooks.
    - Retry logic is delegated to RetryStrategy.
    - Context window management is delegated to ConversationManager.

    Usage::

        loop = AgentLoop(
            model=my_model,
            tools=[search, calculator],
            hooks=registry,
            conversation_manager=manager,
            retry_strategy=exponential_backoff(),
            limits=RunLimits(max_steps=10),
        )

        result = await loop.run("What is the capital of France?")
    """

    def __init__(
        self,
        *,
        model: Any,
        tools: list[Tool | AsyncTool] | None = None,
        system_prompt: str | None = None,
        hooks: HookRegistry | None = None,
        conversation_manager: ConversationManager | None = None,
        retry_strategy: RetryStrategy | None = None,
        limits: RunLimits | None = None,
    ) -> None:
        self._model = model
        self._tools: dict[str, Tool | AsyncTool] = {t.name: t for t in (tools or [])}
        self._system_prompt = system_prompt
        self._hooks = hooks or HookRegistry()
        self._conversation_manager = conversation_manager
        self._retry_strategy = retry_strategy or no_retry()
        self._limits = limits or RunLimits()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        input: str,
        *,
        session_id: str | None = None,
        messages: list[Message] | None = None,
    ) -> AgentResult:
        """
        Run the agent loop for one user input.

        Args:
            input:      The user's message.
            session_id: Optional session identifier passed through to hooks.
            messages:   Prior conversation history to seed the loop with.
                        If omitted the loop starts with an empty history.

        Returns:
            An ``AgentResult`` whether the run succeeded or failed.
        """
        state = RunState()
        history: list[Message] = list(messages or [])

        # Prepend system prompt if provided and not already present
        if self._system_prompt and not any(m.role == "system" for m in history):
            history.insert(0, Message(role="system", content=self._system_prompt))

        # Append the user message
        user_message = Message(role="user", content=input, id=str(uuid.uuid4()))
        history.append(user_message)
        self._hooks.invoke(MessageAdded(message=user_message, step=state.step_count))
        self._hooks.invoke(AgentStarted(input=input, session_id=session_id))

        tool_uses: list[ToolUse] = []
        tool_results: list[ToolResult] = []

        try:
            while True:
                # ── 1. Check limits ────────────────────────────────────────
                exceeded = state.exceeds(self._limits)
                if exceeded is not None:
                    stop_reason = _limit_to_stop_reason(exceeded)
                    return self._build_result(
                        stop_reason=stop_reason,
                        message="",
                        state=state,
                        tool_uses=tool_uses,
                        tool_results=tool_results,
                        success=False,
                        session_id=session_id,
                    )

                # ── 2. Prepare context ─────────────────────────────────────
                if self._conversation_manager is not None:
                    context = self._conversation_manager.prepare(history)
                else:
                    context = list(history)

                # ── 3. Call model ──────────────────────────────────────────
                self._hooks.invoke(
                    BeforeModelCall(messages=context, step=state.step_count)
                )

                model_call_start = time.monotonic()
                response: Message = await self._retry_strategy.execute(
                    lambda: self._invoke_model(context)
                )
                model_call_ms = (time.monotonic() - model_call_start) * 1000

                state.model_call_count += 1
                state.input_tokens += _extract_tokens(response, "input_tokens")
                state.output_tokens += _extract_tokens(response, "output_tokens")

                self._hooks.invoke(
                    AfterModelCall(
                        message=response,
                        step=state.step_count,
                        duration_ms=model_call_ms,
                    )
                )

                history.append(response)
                self._hooks.invoke(
                    MessageAdded(message=response, step=state.step_count)
                )

                # ── 4. Handle tool calls ───────────────────────────────────
                pending_tool_uses = _extract_tool_uses(response)

                if pending_tool_uses:
                    for tool_use in pending_tool_uses:
                        tool_result = await self._execute_tool(
                            tool_use=tool_use,
                            step=state.step_count,
                        )
                        tool_uses.append(tool_use)
                        tool_results.append(tool_result)
                        state.tool_call_count += 1

                        result_message = _tool_result_to_message(tool_result)
                        history.append(result_message)
                        self._hooks.invoke(
                            MessageAdded(
                                message=result_message, step=state.step_count
                            )
                        )

                    state.step_count += 1
                    continue

                # ── 5. Final answer ────────────────────────────────────────
                state.step_count += 1
                result = self._build_result(
                    stop_reason=StopReason.END_TURN,
                    message=response.content,
                    state=state,
                    tool_uses=tool_uses,
                    tool_results=tool_results,
                    success=True,
                    session_id=session_id,
                )
                self._hooks.invoke(
                    AgentFinished(result=result, session_id=session_id)
                )
                return result

        except Exception as error:
            state.step_count += 1
            self._hooks.invoke(
                AgentFailed(
                    error=error, step=state.step_count, session_id=session_id
                )
            )
            return self._build_result(
                stop_reason=StopReason.ERROR,
                message=str(error),
                state=state,
                tool_uses=tool_uses,
                tool_results=tool_results,
                success=False,
                error=error,
                session_id=session_id,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _invoke_model(self, messages: list[Message]) -> Message:
        """Call the model, supporting both sync and async ChatModel."""
        if hasattr(self._model, "invoke"):
            result = self._model.invoke(messages, tools=list(self._tools.values()))
            if asyncio.iscoroutine(result):
                return await result
            return result
        raise TypeError(
            f"Model {type(self._model).__name__!r} does not implement invoke()."
        )

    async def _execute_tool(
        self,
        *,
        tool_use: ToolUse,
        step: int,
    ) -> ToolResult:
        """Look up and execute a tool, returning a ToolResult."""
        self._hooks.invoke(BeforeToolCall(tool_use=tool_use, step=step))

        tool_call_start = time.monotonic()

        tool = self._tools.get(tool_use.name)

        if tool is None:
            tool_result = ToolResult(
                tool_use_id=tool_use.tool_use_id,
                status="error",
                content=f"Tool {tool_use.name!r} is not registered.",
            )
        else:
            try:
                raw = tool.call(tool_use.input)
                if asyncio.iscoroutine(raw):
                    raw = await raw
                tool_result = ToolResult(
                    tool_use_id=tool_use.tool_use_id,
                    status="success",
                    content=json.dumps(raw) if not isinstance(raw, str) else raw,
                )
            except Exception as exc:
                logger.debug(
                    "tool=%s error=%r | tool execution failed", tool_use.name, exc
                )
                tool_result = ToolResult(
                    tool_use_id=tool_use.tool_use_id,
                    status="error",
                    content=str(exc),
                )

        tool_call_ms = (time.monotonic() - tool_call_start) * 1000
        self._hooks.invoke(
            AfterToolCall(
                tool_use=tool_use,
                tool_result=tool_result,
                step=step,
                duration_ms=tool_call_ms,
            )
        )
        return tool_result

    def _build_result(
        self,
        *,
        stop_reason: StopReason,
        message: str,
        state: RunState,
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        success: bool,
        session_id: str | None = None,
        error: Exception | None = None,
    ) -> AgentResult:
        metrics = AgentMetrics(
            steps=state.step_count,
            model_calls=state.model_call_count,
            tool_calls=state.tool_call_count,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            duration_ms=state.elapsed_seconds() * 1000,
        )
        return AgentResult(
            stop_reason=stop_reason,
            message=message,
            metrics=metrics,
            tool_uses=tool_uses,
            tool_results=tool_results,
            success=success,
            error=str(error) if error is not None else None,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _limit_to_stop_reason(limit: str) -> StopReason:
    mapping: dict[str, StopReason] = {
        "max_steps": StopReason.MAX_STEPS,
        "max_model_calls": StopReason.MAX_STEPS,
        "max_tool_calls": StopReason.MAX_STEPS,
        "max_tokens": StopReason.MAX_TOKENS,
        "max_runtime": StopReason.MAX_RUNTIME,
    }
    return mapping.get(limit, StopReason.ERROR)


def _extract_tokens(message: Message, key: str) -> int:
    """Read token counts from message metadata if the model wrote them."""
    usage = message.metadata.get("usage", {})
    return int(usage.get(key, 0)) if isinstance(usage, dict) else 0


def _extract_tool_uses(message: Message) -> list[ToolUse]:
    """
    Extract tool calls from the model response.

    Models report tool calls in message.metadata["tool_calls"] as a list of
    dicts with at minimum: tool_use_id, name, input.
    """
    raw = message.metadata.get("tool_calls")
    if not raw or not isinstance(raw, list):
        return []

    uses: list[ToolUse] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool_use_id = item.get("tool_use_id") or item.get("id") or str(uuid.uuid4())
        name = item.get("name", "")
        input_data = item.get("input") or item.get("arguments") or {}
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                input_data = {"raw": input_data}
        uses.append(ToolUse(tool_use_id=tool_use_id, name=name, input=input_data))
    return uses


def _tool_result_to_message(result: ToolResult) -> Message:
    """Convert a ToolResult into a Message the model can read."""
    return Message(
        role="tool",
        content=result.content,
        id=str(uuid.uuid4()),
        metadata={
            "tool_use_id": result.tool_use_id,
            "status": result.status,
        },
    )
