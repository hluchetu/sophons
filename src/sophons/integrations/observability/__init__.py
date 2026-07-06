from __future__ import annotations

from typing import Any

__all__ = ["LangfuseObservability", "LangfuseSink", "OTELSink"]


# Lazy exports (PEP 562): each sink needs its own optional extra, so nothing
# is imported until first attribute access.
def __getattr__(name: str) -> Any:
    if name in ("LangfuseObservability", "LangfuseSink"):
        from sophons.integrations.observability import langfuse

        return getattr(langfuse, name)
    if name == "OTELSink":
        from sophons.integrations.observability.otel import OTELSink

        return OTELSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
