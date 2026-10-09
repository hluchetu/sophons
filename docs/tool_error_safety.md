# Tool exception safety

Sophons converts exceptions raised during tool execution or result serialization
into an error ToolResult with the original tool-call ID and the fixed message:

> Tool execution failed. The operation could not be completed.

The exception text is not forwarded to the model, AfterToolCall hooks, default
SDK logs, or the tool span's status. The call still counts against the tool budget.
The model can continue responding within the remaining run limits; this does not
automatically retry the tool or undo side effects.

Study `src/sophons/agents/loop.py` and `tests/test_tool_error_safety.py`.
`tests/test_observability.py` also verifies sanitized span status.

This boundary handles caught tool exceptions. Tools still own the safety of
their successful results and any logs/spans they emit themselves. Explicit
guardrail reasons and approval notes are application-controlled content. Model,
hook, and other runtime errors are separate boundaries; this is not a blanket
redaction system. Do not put secrets in tool names or call IDs.
