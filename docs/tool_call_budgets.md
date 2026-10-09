# Tool-call budgets

`RunLimits.max_tool_calls` caps admitted tool-dispatch attempts during one agent
run. Each slot is reserved before dispatch, including calls requested together
in the same model response. The counter starts fresh for each run.

```python
from sophons.agents import Agent
from sophons.agents.state import RunLimits
from sophons.agents.responses import StopReason

agent = Agent(model=model, tools=[search], limits=RunLimits(max_tool_calls=3))
result = await agent.run("Find the relevant provisions")
if result.stop_reason == StopReason.MAX_TOOL_CALLS:
    # The run is incomplete; do not present it as a completed answer.
    ...
```

If a model requests two tools and only one slot remains, only the first admitted
call runs. Remaining requests receive error `ToolResult` entries with their own
call IDs. Those skipped calls do not run their functions or execution hooks and
do not increase `metrics.tool_calls`.

An admitted call consumes its slot even if the tool is missing, a policy denies
it, or the function fails. This prevents failures from becoming unlimited retry
attempts. `tool_uses` records requested calls, while `metrics.tool_calls` counts
admitted attempts; skipped requests can therefore make their lengths differ.

Structured-response tool submissions also consume a slot, preserving the
existing accounting. Invalid structured output cannot retry past the budget.
Output guardrails and schema validation still apply to admitted final results.

Exhaustion terminates the run with `success=False` and `max_tool_calls`; the SDK
does not make an extra model call to synthesize a final answer. Leave headroom
for the final model/structured-response step. A zero budget stops the run before
model work. The value must be a nonnegative integer; booleans and fractional
values are rejected.

This is a per-run dispatch limit, not an application-wide quota, transaction
rollback or complete timeout policy. Model retries and in-flight runtime limits
need separate controls. Bafa's authorization and legal-output policies remain
application responsibilities.

Study `tests/test_tool_call_budget.py`, `RunState.reserve_tool_call` in
`src/sophons/agents/state.py`, and the reservation/skip paths in
`src/sophons/agents/loop.py`.
