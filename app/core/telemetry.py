from __future__ import annotations

import os
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # pragma: no cover - optional dependency for local instrumentation
    trace = None
    OTLPSpanExporter = None
    Status = None
    StatusCode = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    TracerProvider = None
    Resource = None


_provider_configured = False


def configure_telemetry(service_name: str = "triage-agent") -> None:
    """Configure a lightweight OpenTelemetry tracer for local tracing.

    The default behavior emits spans both locally to the console and to Jaeger
    via OTLP HTTP, which makes this a good first production observability step
    without introducing a large framework refactor.
    """
    global _provider_configured

    if trace is None or TracerProvider is None or Resource is None:
        return

    if _provider_configured:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if ConsoleSpanExporter is not None:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if OTLPSpanExporter is not None:
        endpoint = otlp_endpoint or "http://localhost:4318/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    trace.set_tracer_provider(provider)
    _provider_configured = True


def set_span_attributes(span: Any, **attrs: Any) -> None:
    if span is None:
        return

    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, bool):
            span.set_attribute(key, value)
        elif isinstance(value, (int, float)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, str(value))


def record_exception(span: Any, exc: Exception) -> None:
    if span is None or Status is None or StatusCode is None:
        return

    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


tracer = trace.get_tracer("triage-agent") if trace is not None else None
