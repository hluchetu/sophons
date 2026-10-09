# Retry budgets, deadlines and cancellation

Sophons separates three controls:

- `RunLimits.max_model_calls`: admitted attempts of the main agent model across
  the invocation, including failed calls, retries and context-overflow recovery.
- `RetryStrategy.max_attempts`: the ceiling for one model operation, including
  its first try. Its policy decides whether a failure should be retried.
- `RunLimits.max_runtime_seconds`: a monotonic deadline for the agent loop,
  including model/tool work, context preparation, policy checks and retry waits.

```python
import threading
from sophons.agents import Agent, RunLimits, exponential_backoff

cancel = threading.Event()
agent = Agent(
    model=model,
    limits=RunLimits(max_model_calls=4, max_tool_calls=3, max_runtime_seconds=60),
    retry_strategy=exponential_backoff(max_attempts=3),
)
result = await agent.run("Analyze the selected material", cancel_signal=cancel)
# Another task/thread may call cancel.set(). Sophons never clears your event.
```

If the run has two model attempts remaining, a retry policy allowing six cannot
admit a third attempt. Before retry sleeping, Sophons checks the shared model
allowance and whether the backoff fits the remaining deadline. Control stops
are terminal even under a policy that retries all ordinary exceptions.

`max_model_calls` reports admitted model invocations, not successful responses.
It does not count arbitrary requests performed internally by a custom model,
tool, automatic memory extractor or auxiliary summarization model. Disable
provider SDK retries for exact one-invocation/one-request behavior, or account
for those requests in the adapter. Sophons' DeepSeek adapter now defaults to
`sdk_max_retries=0` and an explicit 60-second request timeout; both are configurable.

The caller-owned event returns a failed result with `StopReason.CANCELLED`.
Deadline exhaustion returns `MAX_RUNTIME`, and attempt exhaustion returns
`MAX_MODEL_CALLS`. Cancelling the asyncio task itself propagates `CancelledError`
to the caller rather than pretending the invocation completed.

Blocking synchronous model/tool/context calls run in daemon threads so they do
not block the event loop or its deadline watcher. A late return is discarded.
Stopping the wait does not kill the thread or undo side effects: synchronous
operations need their own bounded I/O timeouts or a process boundary. Async
operations must cooperate with task cancellation; Python cannot forcibly stop
arbitrary non-yielding or cancellation-suppressing code. Synchronous lifecycle
hooks must remain nonblocking.

The deadline currently governs `AgentLoop`, not session loading/saving or
automatic memory work performed by `Agent` outside that loop. Application-wide
request deadlines and persistence failure handling remain separate work.

## Study the implementation

1. `tests/test_retry_deadlines.py`: fake models/tools demonstrating the behavior.
2. `agents/control.py`: invocation-owned cancellation/deadline checks and sync-call isolation.
3. `agents/state.py`: reserve a model slot before each attempt.
4. `agents/loop.py`: apply the shared control around execution and retries.
5. `agents/retry.py`: eligibility/backoff, with a pre-sleep gate and terminal stops.

Design references: Strands' [retry strategies](https://strandsagents.com/docs/user-guide/sdk/agents/retry-strategies/)
and [lifecycle controls](https://strandsagents.com/docs/user-guide/sdk/agents/lifecycle-controls/),
reviewed at commit `79d8d0753a7ae0c324a7c3fde95d714c33122773` in its Apache-2.0-licensed
Python source. The implementation uses Sophons' interfaces and does not vendor
Strands code or add a Strands dependency. Shared attempt-budget enforcement is
a Sophons control rather than a claim about Strands' built-in turn limits.
