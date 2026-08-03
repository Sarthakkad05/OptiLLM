"""
OpenTelemetry Distributed Tracing Setup
Provides tracer provider initialization and context manager for request spans.
"""

import contextlib
import logging
from typing import Any, Dict, Generator, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

logger = logging.getLogger("optillm.core.tracing")

_tracer_initialized = False


def init_tracing(service_name: str = "optillm-gateway"):
    """Initialize OpenTelemetry tracer provider."""
    global _tracer_initialized
    if HAS_OTEL and not _tracer_initialized:
        try:
            provider = TracerProvider()
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            _tracer_initialized = True
            logger.info("Initialized OpenTelemetry tracer provider for service '%s'", service_name)
        except Exception as e:
            logger.warning("Failed to initialize OpenTelemetry tracing: %s", e)


def get_tracer():
    if HAS_OTEL:
        return trace.get_tracer("optillm")
    return None


@contextlib.contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Generator[Any, None, None]:
    """
    Context manager to trace a code block using OpenTelemetry.
    """
    tracer = get_tracer()
    if tracer and HAS_OTEL:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(str(k), str(v))
            yield span
    else:
        yield None
