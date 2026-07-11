from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

_INSTALL_HINT = (
    "The OpenTelemetry SDK is required to export traces. "
    "Install it with: pip install 'sophons[otel]'"
)


class SophonsTelemetry:
    """
    Configures the global OpenTelemetry tracer provider for sophons.

    sophons instruments itself with the OpenTelemetry API, which stays a
    no-op until an SDK tracer provider is registered. This class performs
    that registration and attaches exporters::

        telemetry = SophonsTelemetry()
        telemetry.setup_console_exporter()   # print spans as they finish
        telemetry.setup_otlp_exporter()      # send spans to an OTLP endpoint

    Setup methods return ``self`` so they can be chained.

    The OTLP exporter honours the standard OpenTelemetry environment
    variables (``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``)
    when no explicit ``endpoint`` is given.

    If your application already configures OpenTelemetry globally, pass your
    provider as ``tracer_provider`` (it will not be re-registered), or skip
    this class entirely — sophons picks up the global provider automatically.
    """

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        *,
        service_name: str = "sophons",
    ) -> None:
        if tracer_provider is None:
            tracer_provider = self._create_provider(service_name)
            trace.set_tracer_provider(tracer_provider)
        self._provider = tracer_provider

    @property
    def tracer_provider(self) -> TracerProvider:
        return self._provider

    def setup_console_exporter(self) -> SophonsTelemetry:
        """Print finished spans to stdout. Useful during development."""
        try:
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        self._provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        return self

    def setup_otlp_exporter(self, endpoint: str | None = None) -> SophonsTelemetry:
        """
        Export spans over OTLP/HTTP to any compatible backend
        (Jaeger, Grafana Tempo, Datadog, Langfuse, ...).

        Args:
            endpoint: Base collector URL, e.g. ``http://localhost:4318``.
                      If omitted, the exporter reads the standard
                      ``OTEL_EXPORTER_OTLP_*`` environment variables.
        """
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        if endpoint is not None and not endpoint.endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"

        exporter = OTLPSpanExporter(endpoint=endpoint)
        self._provider.add_span_processor(BatchSpanProcessor(exporter))
        return self

    @staticmethod
    def _create_provider(service_name: str) -> TracerProvider:
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        return TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
