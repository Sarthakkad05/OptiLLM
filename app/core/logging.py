"""
Structured Logging Subsystem.
Supports both standard human-readable text logging (for local development)
and machine-parseable JSON logging (for Datadog, CloudWatch, Loki, ELK in production).
Includes request context tracking via contextvars.
"""

import contextvars
from datetime import datetime, timezone
import json
import logging
import sys
import traceback
from typing import Any, Dict, Optional

from app.core.config import settings

# Context variable to correlate logs with the incoming HTTP request ID
current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_request_id", default=None
)


class JSONFormatter(logging.Formatter):
    """
    Formats log records into a standardized single-line JSON string.
    Fields:
      - timestamp: ISO 8601 UTC timestamp
      - level: Log level (INFO, WARNING, ERROR, etc.)
      - logger: Logger hierarchy name
      - message: Formatted log message
      - request_id: Active request ID if within request context
      - event: Structured event name if provided
      - extra: Arbitrary extra key-value attributes
      - exception: Formatted traceback if exc_info present
    """

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Request ID correlation
        req_id = getattr(record, "request_id", None) or current_request_id.get()
        if req_id:
            log_payload["request_id"] = req_id

        # Event tagging
        event = getattr(record, "event", None)
        if event:
            log_payload["event"] = event

        # Standard LogRecord internal keys to exclude from extra attributes
        standard_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "request_id", "event",
        }

        # Collect extra fields attached via logger.info(..., extra={...})
        extras = {k: v for k, v in record.__dict__.items() if k not in standard_keys}
        if extras:
            # Serialize any non-primitive types safely
            safe_extras = {}
            for k, v in extras.items():
                try:
                    json.dumps(v)
                    safe_extras[k] = v
                except (TypeError, OverflowError):
                    safe_extras[k] = str(v)
            log_payload.update(safe_extras)

        # Exception formatting
        if record.exc_info:
            log_payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        elif record.exc_text:
            log_payload["exception"] = record.exc_text

        return json.dumps(log_payload)


class TextFormatter(logging.Formatter):
    """
    Standard human-readable text formatter for local development console.
    """

    def format(self, record: logging.LogRecord) -> str:
        req_id = getattr(record, "request_id", None) or current_request_id.get()
        req_part = f" [{req_id[:8]}]" if req_id else ""
        asctime = self.formatTime(record, self.datefmt)
        return f"{asctime} | {record.levelname:<8} | {record.name}{req_part} | {record.getMessage()}"


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configures system-wide logging with either text or JSON format."""
    level_str = (log_level or settings.LOG_LEVEL).upper()
    level = getattr(logging, level_str, logging.INFO)
    fmt = (log_format or settings.LOG_FORMAT).lower()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger.addHandler(handler)

    # Silence verbose 3rd-party loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    optillm_logger = logging.getLogger("optillm")
    optillm_logger.info("Logging initialized — level: %s, format: %s", level_str, fmt)
    return optillm_logger


def log_event(
    logger_instance: logging.Logger,
    event: str,
    level: int = logging.INFO,
    message: Optional[str] = None,
    **kwargs: Any,
):
    """
    Convenience helper to emit structured events.
    In JSON format, 'event' and kwargs are top-level JSON fields.
    In text format, they appear appended to the human-readable log line.
    """
    msg = message or f"Event: {event}"
    if kwargs:
        extra_repr = " ".join(f"{k}={v}" for k, v in kwargs.items())
        log_msg = f"{msg} | {extra_repr}"
    else:
        log_msg = msg

    logger_instance.log(level, log_msg, extra={"event": event, **kwargs})


logger = setup_logging()
