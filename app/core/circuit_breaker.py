"""
Circuit Breaker System.
Implements circuit breaker pattern for LLM providers (CLOSED, OPEN, HALF_OPEN).
Prevents cascading failures by auto-removing failing providers from active rotation.
"""

import logging
import time
from enum import Enum
from typing import Any, Dict

from app.core.config import settings

logger = logging.getLogger("optillm.circuit_breaker")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"  # Healthy — traffic flows normally
    OPEN = "OPEN"  # Tripped — provider disabled due to errors
    HALF_OPEN = "HALF_OPEN"  # Recovering — testing limited trial traffic


class ProviderCircuitState:
    """Tracks state and metrics for a single provider."""

    def __init__(self, name: str):
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0.0
        self.last_success_time: float = 0.0
        self.latency_ms: float = 0.0

    def record_success(self, latency_ms: float = 0.0):
        self.failure_count = 0
        self.success_count += 1
        self.last_success_time = time.time()
        if latency_ms > 0:
            # Exponential moving average for smooth latency tracking
            if self.latency_ms == 0.0:
                self.latency_ms = latency_ms
            else:
                self.latency_ms = (self.latency_ms * 0.7) + (latency_ms * 0.3)

        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit Breaker [%s] recovered -> CLOSED", self.name)
            self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if (
            self.state == CircuitState.CLOSED
            and self.failure_count >= settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        ):
            logger.warning(
                "Circuit Breaker [%s] tripped (%d consecutive failures) -> OPEN",
                self.name,
                self.failure_count,
            )
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning(
                "Circuit Breaker [%s] failed during trial -> back to OPEN",
                self.name,
            )
            self.state = CircuitState.OPEN

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            now = time.time()
            if now - self.last_failure_time >= settings.CIRCUIT_BREAKER_RECOVERY_TIME:
                logger.info(
                    "Circuit Breaker [%s] recovery timeout reached -> HALF_OPEN",
                    self.name,
                )
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return True

    def get_status_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "latency_ms": round(self.latency_ms, 2),
            "is_available": self.can_execute(),
        }


class CircuitBreakerManager:
    """Manages circuit breakers for all registered providers."""

    def __init__(self):
        self._circuits: Dict[str, ProviderCircuitState] = {}

    def get_circuit(self, provider_name: str) -> ProviderCircuitState:
        name = provider_name.lower()
        if name not in self._circuits:
            self._circuits[name] = ProviderCircuitState(name)
        return self._circuits[name]

    def record_success(self, provider_name: str, latency_ms: float = 0.0):
        self.get_circuit(provider_name).record_success(latency_ms)

    def record_failure(self, provider_name: str):
        self.get_circuit(provider_name).record_failure()

    def can_execute(self, provider_name: str) -> bool:
        return self.get_circuit(provider_name).can_execute()

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: circuit.get_status_dict() for name, circuit in self._circuits.items()
        }


# Global singleton circuit breaker manager
circuit_breaker = CircuitBreakerManager()
