from __future__ import annotations

from dataclasses import dataclass

from matamaple_trader.domain import OperationalState


@dataclass(frozen=True, slots=True)
class DriftConfig:
    degrade_persistence: int = 3
    pause_persistence: int = 3
    recovery_persistence: int = 3

    def __post_init__(self) -> None:
        if min(self.degrade_persistence, self.pause_persistence, self.recovery_persistence) <= 0:
            raise ValueError("drift persistence values must be positive")


@dataclass(frozen=True, slots=True)
class DriftObservation:
    feature_drift: bool = False
    prediction_drift: bool = False
    calibration_drift: bool = False
    performance_drift: bool = False
    spread_drift: bool = False
    severe: bool = False
    critical_broker_spec: bool = False

    @property
    def breached(self) -> bool:
        return any((self.feature_drift, self.prediction_drift, self.calibration_drift, self.performance_drift, self.spread_drift))


class DriftMonitor:
    """ACTIVE/DEGRADED/PAUSED with persistence; broker-spec critical drift bypasses hysteresis."""

    def __init__(self, config: DriftConfig = DriftConfig()) -> None:
        self.config = config
        self._state: dict[str, OperationalState] = {}
        self._breach_count: dict[str, int] = {}
        self._severe_count: dict[str, int] = {}
        self._recovery_count: dict[str, int] = {}

    def state(self, key: str) -> OperationalState:
        return self._state.get(key, OperationalState.ACTIVE)

    def observe(self, key: str, observation: DriftObservation) -> OperationalState:
        current = self.state(key)
        if observation.critical_broker_spec:
            self._state[key] = OperationalState.DEGRADED
            self._breach_count[key] = 0
            self._severe_count[key] = 0
            self._recovery_count[key] = 0
            return OperationalState.DEGRADED

        if observation.breached:
            self._recovery_count[key] = 0
            self._breach_count[key] = self._breach_count.get(key, 0) + 1
            self._severe_count[key] = self._severe_count.get(key, 0) + (1 if observation.severe else 0)
            if observation.severe and self._severe_count[key] >= self.config.pause_persistence:
                current = OperationalState.PAUSED
            elif self._breach_count[key] >= self.config.degrade_persistence and current is OperationalState.ACTIVE:
                current = OperationalState.DEGRADED
        else:
            self._breach_count[key] = 0
            self._severe_count[key] = 0
            self._recovery_count[key] = self._recovery_count.get(key, 0) + 1
            if self._recovery_count[key] >= self.config.recovery_persistence:
                if current is OperationalState.PAUSED:
                    current = OperationalState.DEGRADED
                    self._recovery_count[key] = 0
                elif current is OperationalState.DEGRADED:
                    current = OperationalState.ACTIVE
                    self._recovery_count[key] = 0
        self._state[key] = current
        return current
