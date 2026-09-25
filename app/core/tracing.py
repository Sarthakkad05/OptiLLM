"""
OpenTelemetry Distributed Tracing Subsystem.
Initializes TracerProvider with Resource attributes and supports both
OTLP exporters (for Jaeger, Tempo, Grafana, Datadog) and fallback exporters.
Provides context managers for gateway lifecycle tracing.
"""

import contextlib
import logging
from typing import Any, Dict, Generator, Optional

from app.core.config import settings

logger = logging.getLogger("optillm.core.tracing")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

_tracer_initialized = False


def init_tracing(service_name: str = "optillm-gateway"):
    """
    Initializes OpenTelemetry tracer provider with OTLP exporter if configured,
    or local console/in-memory provider in development.
    """
    global _tracer_initialized
    if not HAS_OTEL or _tracer_initialized:
        return

    try:
        resource = Resource.create({
            "service.name": service_name,
            "deployment.environment": settings.APP_ENV,
            "service.version": "1.0.0",
        })
        provider = TracerProvider(resource=resource)

        # 1. Check if OTLP endpoint is configured (e.g. Jaeger, Tempo, OpenTelemetry Collector)
        otlp_endpoint = getattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces")
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info("Configured OTLP Trace Exporter pointing to: %s", otlp_endpoint)
            except Exception as exc:
                logger.warning("Failed to configure OTLP Trace Exporter (%s) — falling back to standard processor.", exc)

        # 2. Local console tracing if in development with debug tracing
        elif settings.APP_ENV == "development" and settings.LOG_LEVEL.upper() == "DEBUG":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _tracer_initialized = True
        logger.info("Initialized OpenTelemetry tracer for '%s' (env: %s)", service_name, settings.APP_ENV)
    except Exception as exc:
        logger.warning("Failed to initialize OpenTelemetry tracing: %s", exc)


def get_tracer():
    """Returns the global optillm tracer or None if OTel unavailable."""
    if HAS_OTEL:
        if not _tracer_initialized:
            init_tracing()
        return trace.get_tracer("optillm")
    return None


@contextlib.contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Generator[Any, None, None]:
    """
    Context manager to trace a block of execution with OpenTelemetry.
    Attaches attributes to the span and records exceptions if raised.
    """
    tracer = get_tracer()
    if tracer and HAS_OTEL:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        # OTel attributes must be primitives or sequence of primitives
                        if isinstance(v, (str, bool, int, float)):
                            span.set_attribute(k, v)
                        else:
                            span.set_attribute(k, str(v))
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise
    else:
        yield None
