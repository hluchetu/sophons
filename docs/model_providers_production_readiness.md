# Model Provider Production-Readiness Improvements

## Purpose

Sophons supports provider-neutral model contracts with provider-specific adapters. A successful request path is only the first implementation milestone. A model provider should be considered production-ready only when its supported behavior is clearly defined, validated, tested, observable, secure, and resilient.

This document applies to all model integrations, including Bedrock, Anthropic, DeepSeek, Ollama, OpenAI-compatible providers, and future providers.

## Design principle

Applications should depend on Sophons contracts rather than provider SDKs:

```text
Application or agent
        ↓
Sophons ChatModel / AsyncChatModel
        ↓
Provider adapter
        ↓
Provider SDK and API
```

Provider-specific request formats, response formats, credentials, exceptions, and workarounds must remain inside the provider integration.

## Existing foundation

The current model layer already establishes several important foundations:

- Provider-neutral `Message` objects.
- Provider-neutral `ChatModel` and `AsyncChatModel` protocols.
- Tool definitions independent of a specific model vendor.
- Provider adapters for translating messages and tools.
- Structured output through the tool-calling channel.
- Agent retry and lifecycle machinery.
- Lazy loading of optional provider dependencies.
- Public imports through `sophons.models`.

## Production-readiness requirements

### 1. Provider-neutral exceptions

Provider SDK exceptions should not leak into applications. Sophons should define a stable exception hierarchy such as:

```text
ModelError
├── ModelAuthenticationError
├── ModelAuthorizationError
├── ModelThrottledError
├── ModelTimeoutError
├── ModelUnavailableError
├── ModelInvalidRequestError
├── ContextWindowExceededError
├── StructuredOutputError
└── ModelResponseError
```

Each adapter should translate provider errors into the most appropriate Sophons exception while preserving the original exception as the cause.

Applications and retry policies should depend on Sophons exceptions rather than AWS, Anthropic, OpenAI, or other provider error classes.

### 2. Retry classification

Adapters must distinguish retryable failures from permanent failures.

Potentially retryable:

- Throttling
- Temporary service unavailability
- Model-not-ready responses
- Connection resets
- Some timeouts

Normally not retryable:

- Invalid credentials
- Missing permissions
- Invalid model identifiers
- Unsupported request fields
- Schema validation failures
- Requests exceeding the context window

Retries should use bounded exponential backoff with jitter. They must respect cancellation and must not retry indefinitely.

### 3. Explicit timeouts

Provider clients should have explicit limits for:

- Connection establishment
- Response reading
- Total request duration where supported
- Streaming inactivity

Timeout values should be configurable, documented, and given safe defaults.

### 4. Request validation

Validate requests before sending them to a provider:

- Model identifiers are non-empty.
- At least one usable conversation message exists.
- Only supported roles are present.
- Messages do not produce invalid empty content blocks.
- Tool names and schemas satisfy provider constraints.
- Tool-use IDs are present.
- Tool results reference valid tool-use IDs.
- Token and sampling parameters are within valid ranges.
- Provider-specific combinations of parameters are permitted.

Early validation produces clearer errors and avoids unnecessary API calls and cost.

### 5. Response validation

Treat provider responses as untrusted external data.

Adapters should verify:

- Required response sections exist.
- Content blocks have recognized types.
- Tool calls contain IDs, names, and valid input objects.
- Token-usage fields are safely normalized.
- Stop reasons are recognized or preserved.
- Malformed structured output produces an explicit error.

Unknown response fields should be preserved in metadata when useful but should not break stable Sophons contracts.

### 6. Structured-output reliability

Structured output should not depend only on prompting.

Sophons should define a provider-neutral mechanism for expressing tool choice or required structured output. Provider adapters can then map it to capabilities such as:

- Required tool use
- Forced selection of a named tool
- Native JSON-schema responses
- Strict tool schemas
- Tool-based structured output when native schemas are unavailable

All structured results must be validated using the requested Pydantic model. Validation failures should use bounded correction attempts and produce a clear `StructuredOutputError` when exhausted.

### 7. Tool-use correctness

Every provider adapter should support and test the complete cycle:

```text
Tool definition
→ model tool request
→ Sophons tool execution
→ tool result message
→ model continuation
→ final answer
```

Adapters must correctly handle:

- Multiple tools
- Multiple tool calls in one turn
- Consecutive tool results
- Tool errors
- Invalid tool names
- Invalid arguments
- Duplicate or missing tool-use IDs
- Provider-specific role and content-block requirements
- Maximum-step and termination rules

### 8. Sync, async, and streaming behavior

Sophons should clearly distinguish:

- Synchronous invocation
- Asynchronous invocation
- Streaming invocation

Blocking provider SDK calls must not block an asynchronous event loop. If a provider SDK is synchronous, the async adapter should use an appropriate worker-thread boundary or a supported asynchronous client.

Streaming should have a provider-neutral event contract covering:

- Text deltas
- Tool-call deltas
- Message completion
- Usage metadata
- Stop reasons
- Provider errors
- Cancellation

### 9. Observability

Model calls should expose consistent telemetry:

- Provider
- Model identifier
- Request duration
- Input tokens
- Output tokens
- Cache-read and cache-write tokens
- Stop reason
- Retry count
- Tool-call count
- Error category
- Optional cost estimate

OpenTelemetry spans and metrics should use consistent semantic attributes across providers.

Logging must avoid exposing:

- API keys and credentials
- Sensitive prompts or responses by default
- Personal or regulated information
- Raw tool results containing confidential data

Prompt or response logging should be explicit, configurable, and safely redacted.

### 10. Configuration and credentials

Provider adapters should support secure runtime configuration without requiring credentials in source code.

Requirements include:

- Environment or workload-based credentials
- Managed identities and IAM roles where available
- Optional explicit clients for tests and advanced use cases
- Region and endpoint configuration where applicable
- Private or custom endpoints
- Clear precedence rules for configuration
- Safe configuration representation that does not expose secrets

### 11. Security controls

The model layer should support or integrate with:

- Least-privilege provider permissions
- Approved-model allowlists
- Region restrictions
- Input and output guardrails
- Sensitive-data filtering
- Tool authorization
- Audit events
- Tenant isolation
- Data-retention and logging policies

Provider compliance does not automatically make an application compliant. Applications remain responsible for their configuration and data flows.

### 12. Cost and resource controls

Support safeguards such as:

- Maximum input and output tokens
- Request and concurrency limits
- Per-run tool-step limits
- Model allowlists
- Usage accounting
- Budget alerts at the deployment layer
- Optional caching
- Protection against unbounded retries and agent loops

### 13. Capability discovery

Providers and models differ. Sophons should avoid assuming every model supports every feature.

A future capability model could describe:

- Tool use
- Required tool choice
- Strict schemas
- Native structured output
- Streaming
- Vision and other media
- Prompt caching
- Token counting
- Guardrails
- Maximum context size

Unsupported capabilities should fail clearly before or during request construction rather than silently degrading behavior.

### 14. Testing strategy

Every provider should have contract tests for:

- Public import behavior
- Optional dependency loading
- Basic text invocation
- System-message handling
- Multi-turn conversation serialization
- Tool-definition serialization
- Tool-call normalization
- Tool-result serialization
- Multiple tool calls
- Token-usage normalization
- Stop-reason handling
- Invalid and incomplete responses
- Error translation
- Timeouts and retry classification
- Structured-output validation

Unit tests should inject fake clients and make no network calls.

Integration tests should:

- Be explicitly marked
- Require opt-in credentials
- Use a low-cost model and bounded token limits
- Avoid sensitive data
- Verify one real request and, where supported, one tool-use cycle
- Never run automatically in ordinary pull-request workflows

### 15. Documentation requirements

Each provider should document:

- Installation extra
- Basic usage
- Authentication
- Supported configuration
- Supported capabilities
- Known limitations
- Required permissions
- Error behavior
- Sync, async, and streaming support
- Safe production deployment guidance

## Recommended implementation phases

### Phase 1: Correct vertical slice

- Basic text invocation
- System and conversation messages
- Tool definitions and tool calls
- Tool results
- Response and usage normalization
- Injectable fake client
- Public import
- Unit tests

### Phase 2: Reliability

- Provider-neutral exception hierarchy
- Error translation
- Retry classification
- Explicit timeouts
- Request and response validation
- Structured-output enforcement

### Phase 3: Operations

- OpenTelemetry integration
- Safe logging and redaction
- Usage and cost metrics
- Cancellation and concurrency behavior
- Integration-test suite

### Phase 4: Advanced capabilities

- Streaming
- Multimodal content
- Guardrails
- Prompt caching
- Native token counting
- Service tiers
- Provider-specific advanced features

## Definition of done for a provider

A provider is production-ready for its documented feature set when:

- [ ] It satisfies the public Sophons model protocol.
- [ ] Its optional dependency loads lazily.
- [ ] Requests and responses are validated.
- [ ] Provider errors are translated into Sophons exceptions.
- [ ] Retryable failures are classified correctly.
- [ ] Timeouts are explicit.
- [ ] Tool use works through a complete multi-turn cycle.
- [ ] Structured output is enforced and validated.
- [ ] Usage and stop metadata are normalized.
- [ ] Sensitive data is not logged by default.
- [ ] Unit tests make no network calls.
- [ ] Opt-in integration tests verify the real provider.
- [ ] Configuration, permissions, capabilities, and limitations are documented.
- [ ] The supported behavior has monitoring guidance.

## Guiding principle

Production readiness is not measured by implementation size. It means the provider's promised behavior is explicit, tested, secure, resilient, and observable. Sophons should add advanced capabilities only through stable provider-neutral contracts rather than leaking vendor-specific details into applications.
