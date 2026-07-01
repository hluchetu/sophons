from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from sophons.agents.conversation import ConversationManager
from sophons.agents.hooks import HookRegistry
from sophons.agents.loop import AgentLoop
from sophons.agents.responses import AgentResult
from sophons.agents.retry import RetryStrategy, exponential_backoff
from sophons.agents.session import SessionManager
from sophons.agents.state import RunLimits
from sophons.tools.base import AsyncTool, Tool

logger = logging.getLogger(__name__)


class Agent:
    """
    The public entry point for running a Sophons agent.

    ``Agent`` wires together all the building blocks — model, tools, hooks,
    conversation management, session persistence, and retry — into one object
    you can call from your application.

    Minimal usage::

        agent = Agent(model=my_model)
        result = await agent.run("What is the capital of France?")

    With everything configured::

        agent = Agent(
            model=my_model,
            tools=[search, calculator],
            system_prompt="You are a helpful assistant.",
            hooks=registry,
            conversation_manager=TokenBudgetManager(max_tokens=4096, token_counter=tc),
            session_manager=FileSessionManager("./sessions"),
            retry_strategy=exponential_backoff(),
            limits=RunLimits(max_steps=10),
        )

        result = await agent.run("Hello", session_id="user-123")

    Session behaviour:
        If ``session_manager`` is provided and ``session_id`` is passed to
        ``run()``, the agent loads prior conversation history before the run
        and saves the updated history after — regardless of whether the run
        succeeded or failed.

    Sync usage::

        result = agent.run_sync("Hello")
    """

    def __init__(
        self,
        *,
        model: Any,
        tools: list[Tool | AsyncTool] | None = None,
        system_prompt: str | None = None,
        hooks: HookRegistry | None = None,
        conversation_manager: ConversationManager | None = None,
        session_manager: SessionManager | None = None,
        retry_strategy: RetryStrategy | None = None,
        limits: RunLimits | None = None,
    ) -> None:
        self._session_manager = session_manager
        self._loop = AgentLoop(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            hooks=hooks,
            conversation_manager=conversation_manager,
            retry_strategy=retry_strategy or exponential_backoff(),
            limits=limits,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        input: str,
        *,
        session_id: str | None = None,
    ) -> AgentResult:
        """
        Run the agent for one user message.

        Args:
            input:      The user's message.
            session_id: When provided and a ``session_manager`` is configured,
                        conversation history is loaded before the run and saved
                        after.  Pass the same ID across calls to maintain a
                        continuous conversation.  If no ID is given a fresh
                        conversation is started each time.

        Returns:
            ``AgentResult`` with the final message, stop reason, metrics,
            and any tool activity from this run.
        """
        prior_messages = await self._load_session(session_id)

        result = await self._loop.run(
            input,
            session_id=session_id,
            messages=prior_messages,
        )

        await self._save_session(session_id, prior_messages, input, result)
        return result

    def run_sync(
        self,
        input: str,
        *,
        session_id: str | None = None,
    ) -> AgentResult:
        """
        Synchronous wrapper around ``run()``.

        Useful for scripts, REPLs, and tests that are not running inside an
        async event loop.  Cannot be called from inside a running event loop —
        use ``await agent.run()`` there instead.

        Raises:
            RuntimeError: If called from inside a running event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "Agent.run_sync() cannot be called from inside a running event loop. "
                "Use 'await agent.run()' instead."
            )

        return asyncio.run(self.run(input, session_id=session_id))

    def new_session_id(self) -> str:
        """Generate a fresh unique session ID."""
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    async def _load_session(self, session_id: str | None) -> list:
        if self._session_manager is None or session_id is None:
            return []
        try:
            messages = await self._session_manager.load(session_id)
            logger.debug(
                "session=%s loaded %d prior messages", session_id, len(messages)
            )
            return messages
        except Exception as exc:
            logger.warning(
                "session=%s failed to load history: %r — starting fresh",
                session_id,
                exc,
            )
            return []

    async def _save_session(
        self,
        session_id: str | None,
        prior_messages: list,
        user_input: str,
        result: AgentResult,
    ) -> None:
        if self._session_manager is None or session_id is None:
            return

        from sophons.models.messages import Message

        updated = list(prior_messages)
        updated.append(Message(role="user", content=user_input))
        if result.message:
            updated.append(Message(role="assistant", content=result.message))

        try:
            await self._session_manager.save(session_id, updated)
            logger.debug(
                "session=%s saved %d messages", session_id, len(updated)
            )
        except Exception as exc:
            logger.warning(
                "session=%s failed to save history: %r", session_id, exc
            )
