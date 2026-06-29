from __future__ import annotations

from sophons.agents.agent import Agent
from sophons.agents.conversation import (
    ConversationManager,
    ContextBudgetExceededError,
    ManagerPipeline,
    SlidingWindowManager,
    SummarizationError,
    SummarizingManager,
    TokenBudgetManager,
    TokenCounter,
    ToolInteractionCompactor,
)
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
from sophons.agents.loop import AgentLoop
from sophons.agents.responses import (
    AgentMetrics,
    AgentResult,
    StopReason,
    ToolResult,
    ToolUse,
)
from sophons.agents.retry import (
    BackoffSettings,
    RetryContext,
    RetryDecision,
    RetryPolicy,
    RetryStrategy,
    all_of,
    any_of,
    exponential_backoff,
    never,
    no_retry,
    on_exception,
    on_status_codes,
)
from sophons.agents.session import (
    FileSessionManager,
    InMemorySessionManager,
    SessionManager,
)
from sophons.agents.state import RunLimits, RunState

__all__ = [
    # Agent
    "Agent",
    "AgentLoop",
    # Responses
    "AgentMetrics",
    "AgentResult",
    "StopReason",
    "ToolUse",
    "ToolResult",
    # Hooks
    "HookRegistry",
    "AgentStarted",
    "AgentFinished",
    "AgentFailed",
    "BeforeModelCall",
    "AfterModelCall",
    "BeforeToolCall",
    "AfterToolCall",
    "MessageAdded",
    # State
    "RunLimits",
    "RunState",
    # Conversation
    "ConversationManager",
    "TokenCounter",
    "SlidingWindowManager",
    "TokenBudgetManager",
    "ToolInteractionCompactor",
    "SummarizingManager",
    "ManagerPipeline",
    "ContextBudgetExceededError",
    "SummarizationError",
    # Retry
    "RetryStrategy",
    "RetryPolicy",
    "RetryContext",
    "RetryDecision",
    "BackoffSettings",
    "no_retry",
    "exponential_backoff",
    "never",
    "on_exception",
    "on_status_codes",
    "any_of",
    "all_of",
    # Session
    "SessionManager",
    "InMemorySessionManager",
    "FileSessionManager",
]
