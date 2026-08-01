import time

from app.core.circuit_breaker import CircuitBreakerManager, CircuitState
from app.core.config import settings


def test_circuit_breaker_initial_state():
    cb = CircuitBreakerManager()
    state = cb.get_circuit("openai")

    assert state.state == CircuitState.CLOSED
    assert state.can_execute() is True
    assert state.failure_count == 0


def test_circuit_breaker_trips_to_open():
    cb = CircuitBreakerManager()
    provider_name = "test_provider"

    # Record threshold failures
    for _ in range(settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        cb.record_failure(provider_name)

    circuit = cb.get_circuit(provider_name)
    assert circuit.state == CircuitState.OPEN
    assert circuit.can_execute() is False


def test_circuit_breaker_recovery_to_half_open():
    cb = CircuitBreakerManager()
    provider_name = "recovery_provider"

    for _ in range(settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        cb.record_failure(provider_name)

    circuit = cb.get_circuit(provider_name)
    assert circuit.state == CircuitState.OPEN

    # Simulate recovery time passing
    circuit.last_failure_time = time.time() - (
        settings.CIRCUIT_BREAKER_RECOVERY_TIME + 1.0
    )

    assert circuit.can_execute() is True
    assert circuit.state == CircuitState.HALF_OPEN

    # Record success to close circuit
    cb.record_success(provider_name, latency_ms=50.0)
    assert circuit.state == CircuitState.CLOSED
    assert circuit.latency_ms > 0.0
