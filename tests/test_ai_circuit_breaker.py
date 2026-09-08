import pytest

from matamaple_trader.ai import AICircuitBreaker
from matamaple_trader.config import RuntimeConfig


def test_circuit_opens_after_consecutive_failures_and_recovers_after_cooldown():
    breaker = AICircuitBreaker(max_failures=2, cooldown_seconds=10.0)
    assert breaker.allow_request(now=100.0)
    breaker.record_failure(now=100.0)
    assert breaker.allow_request(now=101.0)
    breaker.record_failure(now=102.0)
    assert not breaker.allow_request(now=105.0)
    state = breaker.state(now=105.0)
    assert state.open
    assert state.consecutive_failures == 2
    assert state.retry_after_seconds == pytest.approx(7.0)
    assert breaker.allow_request(now=112.0)
    assert not breaker.state(now=112.0).open


def test_success_resets_circuit():
    breaker = AICircuitBreaker(max_failures=2, cooldown_seconds=10.0)
    breaker.record_failure(now=1.0)
    breaker.record_success()
    assert breaker.allow_request(now=2.0)
    assert breaker.state(now=2.0).consecutive_failures == 0


def test_runtime_config_rejects_non_loopback_ai_endpoint():
    with pytest.raises(ValueError):
        RuntimeConfig(ai_base_url="http://localhost.evil.com:11434")


def test_runtime_config_rejects_embedded_ai_credentials():
    with pytest.raises(ValueError):
        RuntimeConfig(ai_base_url="http://user:pass@localhost:11434")


def test_runtime_config_keeps_live_trading_disabled():
    cfg = RuntimeConfig()
    assert cfg.execution_mode == "DRY_RUN"
    assert cfg.live_trading_enabled is False


def test_ai_latency_cannot_exceed_timeout():
    with pytest.raises(ValueError):
        RuntimeConfig(ai_timeout_seconds=5.0, ai_max_latency_seconds=10.0)
