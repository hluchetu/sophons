# Context Management Architecture — Design Notes

This note is a local design guide for how Sophons decides what a model sees.
It records a problem with the current shape, the proposed replacement, and
the reasoning for each divergence from other SDKs.

Status: **proposal**. Nothing here is implemented yet.

## Core Idea

Conversation history grows without bound; the model's context window does
not. Something must decide what to keep.

```mermaid
flowchart LR
    History["full history"] --> Manager["conversation manager"]
    Manager --> Context["what the model sees"]
    Context --> Model["model"]
    Model -->|overflow error| Reduce["reduce_context"]
    Reduce --> Context
```

There are exactly three answers to "what do we keep":

1. everything
2. the recent part
3. the recent part, plus a summary of the rest

Everything else — how you measure size, how you avoid splitting a tool call
from its result, how you shrink an oversized tool result — is machinery in
service of those three, not a fourth answer.

## The problem with the current shape

Sophons exposes six manager classes plus a counter protocol:

```
NullConversationManager
SlidingWindowManager
TokenBudgetManager
ToolInteractionCompactor
SummarizingManager
ManagerPipeline
```

Only three of those are strategies. The rest are mechanisms that leaked into
the strategy namespace:

- **`TokenBudgetManager` is not a strategy.** It is a sliding window measured
  in tokens rather than messages. That is a parameter.
- **`ToolInteractionCompactor` is not a strategy.** It shrinks tool traffic —
  something every manager should do *before* it resorts to discarding whole
  messages.
- **`ManagerPipeline` exists only to recombine what should not have been
  split.** It is a symptom of the decomposition, not a feature.

The most serious consequence: **correctness became a strategy choice.**
`SlidingWindowManager` can return a tool result whose originating tool call
was dropped — an orphan that some providers reject outright.
`TokenBudgetManager` cannot, because it groups tool interactions into atomic
units. So picking a strategy silently picks whether the output is well-formed.
`tests/test_conversation.py::test_sliding_window_can_split_a_tool_pair`
currently documents that as behaviour. It is a bug.

For comparison, Strands exposes three managers and two methods, and folds
tool-pair preservation and tool-result truncation *into* the sliding window
rather than offering them as alternatives to it.

## Proposed shape

```mermaid
flowchart TD
    Protocol["ConversationManager (protocol)<br/>prepare · reduce_context"]
    Protocol --> Null["NullConversationManager"]
    Protocol --> Window["SlidingWindowManager<br/>max_messages | max_tokens"]
    Protocol --> Summary["SummarizingManager<br/>compression_threshold"]

    Shared["shared internals:<br/>tool-pair grouping<br/>tool-result truncation<br/>token counting"]
    Window -.uses.-> Shared
    Summary -.uses.-> Shared
```

Three public strategies:

```python
NullConversationManager()

SlidingWindowManager(
    max_messages: int | None = None,
    max_tokens: int | None = None,
    token_counter: TokenCounter | None = None,   # defaults when max_tokens set
    truncate_tool_results: bool = True,
    preserve_system_messages: bool = True,
)

SummarizingManager(
    model: ChatModel,
    keep_recent_messages: int,
    compression_threshold: float | None = None,  # fraction of the window
    trigger_message_count: int | None = None,
    trigger_token_count: int | None = None,
    token_counter: TokenCounter | None = None,
)
```

Two protocol methods:

```python
def prepare(messages, context) -> list[Message]: ...
def reduce_context(messages, context, error) -> list[Message]: ...
```

### What moves where

| Today | Becomes |
|---|---|
| `TokenBudgetManager(max_tokens=n, ...)` | `SlidingWindowManager(max_tokens=n)` |
| `ToolInteractionCompactor(keep_recent=n)` | `truncate_tool_results=True` (default, applied before dropping) |
| `ManagerPipeline([...])` | deleted — nothing left to recombine |
| tool-pair grouping in `TokenBudgetManager` | shared internal, used by every manager, always on |
| `ApproximateTokenCounter` as a required argument | the default whenever a token limit is set |

## Decisions and their reasons

**Tool-pair preservation is an invariant, not an option.** Sending a tool
result without its call is never what anyone wants. It moves into a shared
private helper and stops being reachable as a configuration.

**Truncation runs before dropping.** Shrinking a 10 KB tool result costs
nothing in meaning; dropping the message that contains it costs the agent its
record of what it did. Ordering these correctly is the point of merging them.

**`prepare()` stays pure — a deliberate divergence.** Strands mutates the
agent's message list in `apply_management`. Sophons returns a view. Purity is
why these managers are a few lines each and testable without constructing an
agent, and it keeps the full history intact for sessions and observability.
The cost is real and is addressed below.

**`reduce_context()` is borrowed from Strands.** An agent that exceeds its
window currently raises an unhandled provider error. Catching a typed
overflow, letting the manager reduce, and retrying is the difference between
degrading and failing. This is convergent design: any SDK that can overflow
needs it.

**A ratio beats an absolute threshold.** `compression_threshold=0.7` means
"compress past 70% of this model's window", which the library derives from
`model.context_window`. An absolute token count pushes onto the caller a
number they cannot pick correctly. Also from Strands, whose
`proactive_compression` expresses the same idea for the same reason.

**Token counting is a default, never a requirement.** A protocol with no
implementation meant `TokenBudgetManager` could not be constructed from the
library alone. Any manager with a token limit and no counter uses
`ApproximateTokenCounter`.

## Known gaps this does not close

**Compression is recomputed, not persisted.** Because `prepare()` is pure and
`AgentLoop` only ever appends to `history`, a summarizing manager re-summarizes
on every model call — three steps past the threshold means three summarization
calls over near-identical history
(`test_summarizing_recomputes_on_every_call`). The summary-merging path in
`SummarizingManager` is likewise unreachable through `AgentLoop`, since no
summary is ever fed back in as history.

Fixing this means the loop caching a manager's output against the history it
covered, so a summary is computed once and reused until the history changes
beneath it. That preserves purity — the manager stays a function; the *loop*
gains a cache — and is a separate piece of work from this refactor.

**Messages without ids escape the token budget.** The current filter is
`m.id in kept_ids or m.id is None`, so any message lacking an id bypasses the
limit entirely; model adapters return assistant messages with no id
(`test_token_budget_always_keeps_messages_without_ids`). The merged window
manager should select by position rather than by id set.

## Migration

Nothing is published and the only consumer is `sophons-examples`, so this is
the cheapest this change will ever be.

1. Add `reduce_context` to the protocol with a default implementation.
2. Merge `TokenBudgetManager` into `SlidingWindowManager` as `max_tokens`.
3. Move tool-pair grouping into a shared helper; apply it in every manager.
4. Fold `ToolInteractionCompactor` into `truncate_tool_results`.
5. Delete `ManagerPipeline`.
6. Rewrite the three tests that currently pin defective behaviour —
   `test_sliding_window_can_split_a_tool_pair` and
   `test_token_budget_always_keeps_messages_without_ids` should invert, since
   the refactor is what makes them wrong.
7. Update `memory/context_window.py`, which is still unwritten and should be
   written against the new shape rather than the old one.
