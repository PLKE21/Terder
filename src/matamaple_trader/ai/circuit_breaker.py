from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class CircuitState:
    open: bool
    consecutive_failures: int
    retry_after_seconds: float


class AICircuitBreaker:
    """Small in-process circuit breaker for the local AI reviewer."""

    def __init__(self, *, max_failures: int = 3, cooldown_seconds: float = 60.0) -> None:
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None

    def state(self, *, now: float | None = None) -> CircuitState:
        now = monotonic() if now is None else now
        if self._opened_at is None:
            return CircuitState(False, self._failures, 0.0)
        elapsed = now - self._opened_at
        remaining = self.cooldown_seconds - elapsed
        if remaining <= 0:
            return CircuitState(False, self._failures, 0.0)
        return CircuitState(True, self._failures, remaining)

    def allow_request(self, *, now: float | None = None) -> bool:
        now = monotonic() if now is None else now
        if self._opened_at is None:
            return True
        if now - self._opened_at >= self.cooldown_seconds:
            self._opened_at = None
            self._failures = 0
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self, *, now: float | None = None) -> None:
        self._failures += 1
        if self._failures >= self.max_failures:
            self._opened_at = monotonic() if now is None else now
