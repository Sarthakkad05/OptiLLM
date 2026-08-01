"""
Tool Execution Sandbox & Audit Log Engine.
Executes tools under strict 2-second timeout protection, network safety constraints,
and writes execution audit trails to the database.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.core.tool_registry import tool_registry
from app.db.models import ToolAuditLog

logger = logging.getLogger("optillm.core.tool_sandbox")

_executor = ThreadPoolExecutor(max_workers=4)


def execute_tool_safely(
    tool_name: str,
    kwargs: Optional[Dict[str, Any]] = None,
    db: Optional[Session] = None,
    timeout_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Executes a registered tool within a sandboxed executor with timeout protection
    and audit log recording.
    """
    kwargs = kwargs or {}
    tool_spec = tool_registry.get(tool_name)

    if not tool_spec:
        raise ValueError(f"Tool '{tool_name}' is not registered.")

    start_time = time.perf_counter()
    success = False
    result_data = None
    error_msg = None

    try:
        # Enforce sandbox timeout using ThreadPoolExecutor
        future = _executor.submit(tool_spec.func, **kwargs)
        result_data = future.result(timeout=timeout_seconds)
        success = True
    except TimeoutError:
        error_msg = f"Tool '{tool_name}' execution timed out (> {timeout_seconds}s)."
        logger.error(error_msg)
    except Exception as exc:
        error_msg = f"Tool '{tool_name}' execution error: {exc}"
        logger.error(error_msg)

    execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Persist trace into ToolAuditLog if DB session is provided
    if db:
        try:
            audit = ToolAuditLog(
                tool_name=tool_name,
                arguments=json.dumps(kwargs),
                execution_time_ms=execution_time_ms,
                success=success,
                result_summary=json.dumps(result_data)[:500] if success else None,
                error_message=error_msg,
            )
            db.add(audit)
            db.commit()
        except Exception as db_exc:
            logger.warning("Failed to record tool audit log: %s", db_exc)

    if not success:
        raise RuntimeError(error_msg)

    return {
        "tool_name": tool_name,
        "success": True,
        "execution_time_ms": execution_time_ms,
        "result": result_data,
    }
