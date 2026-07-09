"""Attribute names used on sophons spans.

Follows the OpenTelemetry GenAI semantic conventions where one exists;
everything else lives under the ``sophons.`` namespace.
"""

# ── GenAI semantic conventions ─────────────────────────────────────────────────

REQUEST_MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"

# ── sophons namespace ──────────────────────────────────────────────────────────

SESSION_ID = "sophons.session_id"
STEP = "sophons.step"
STEPS = "sophons.steps"
MODEL_CALLS = "sophons.model_calls"
TOOL_CALLS = "sophons.tool_calls"
CACHE_READ_TOKENS = "sophons.cache_read_tokens"
CACHE_WRITE_TOKENS = "sophons.cache_write_tokens"
STOP_REASON = "sophons.stop_reason"

LOADER = "sophons.loader"
SPLITTER = "sophons.splitter"
RETRIEVER = "sophons.retriever"
PATH = "sophons.path"
URL = "sophons.url"
DOCUMENT_COUNT = "sophons.document_count"
CHUNK_COUNT = "sophons.chunk_count"
RESULT_COUNT = "sophons.result_count"
LIMIT = "sophons.limit"
NAMESPACE = "sophons.namespace"
ENTRY_COUNT = "sophons.entry_count"
INVALIDATED_COUNT = "sophons.invalidated_count"
