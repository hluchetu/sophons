# Session persistence outcomes

Session storage is explicit: `load` returns an empty list only for a genuinely
missing session. Corrupt records, unreadable files and repository failures raise
`SessionPersistenceError` with a safe operation message. The original error is
chained for controlled developer diagnostics, not printed in SDK logs/results.

```python
from sophons.agents import Agent, StopReason

result = await agent.run("Continue the conversation", session_id="owned-session")
if result.stop_reason == StopReason.SESSION_ERROR:
    # Treat the turn as failed; don't display it as a saved successful answer.
    print(result.message)
```

If loading fails, Agent stops before model work. If saving fails, it returns
`success=False`, `stop_reason=session_error`, a safe error message, and no final
typed output. Completed work metrics and tool diagnostics are retained. The
SDK does not regenerate the answer automatically. A save failure does not undo
tools already executed; applications need idempotency before retrying a turn.

`AgentFinished` fires only after successful session saving and post-run memory
work. A persistence error instead emits `AgentFailed` carrying a safe
`SessionPersistenceError`. Direct `AgentLoop` calls still signal completion of
their own loop; that is not proof that application storage committed a result.

`FileSessionManager` serializes to a private temporary file in the same
directory, flushes/fsyncs that file and atomically replaces the session JSON.
Failure before replacement preserves the previous history. Temporary files
are cleaned up; corruption is never overwritten by silently starting fresh.

Atomic replacement is not concurrency control or a database transaction.
Concurrent writers remain last-writer-wins, and complete power-loss durability
depends on the filesystem. A custom repository can fail after an ambiguous
commit; Agent reports failure rather than promising rollback. Cancellation of
a blocking file operation also cannot guarantee that an already-started write
was undone. Use revision/idempotency checks in application-owned storage.

The existing agent-loop deadline does not cover session loading/saving or
automatic memory operations outside the loop. Applications needing an overall
request/storage timeout must apply that boundary separately. Firm/matter access,
history retention and shared-conversation policy remain application rules.

Study `tests/test_session_failures.py`, the Agent session helpers in
`agents/agent.py`, `agents/session.py`, and `SessionPersistenceError` in `errors.py`.
