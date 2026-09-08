from __future__ import annotations

from dataclasses import dataclass

from matamaple_trader.domain import OperationalState, SignalResult


@dataclass(frozen=True, slots=True)
class SignalRow:
    symbol: str
    signal: str
    probability: float | None
    regime: str
    grade: str
    risk_reward: float | None
    spread_state: str
    operational_state: str
    pipeline_version: str
    warning: str | None


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    rows: tuple[SignalRow, ...]
    active: int
    degraded: int
    paused: int


def build_dashboard_snapshot(results: list[SignalResult] | tuple[SignalResult, ...]) -> DashboardSnapshot:
    rows = tuple(
        SignalRow(
            symbol=r.symbol,
            signal=r.signal.value,
            probability=r.calibrated_probability,
            regime=r.regime,
            grade=r.grade,
            risk_reward=r.risk_reward,
            spread_state=r.spread_state,
            operational_state=r.operational_state.value,
            pipeline_version=r.pipeline_version,
            warning=r.portfolio_warning,
        )
        for r in results
    )
    return DashboardSnapshot(
        rows=rows,
        active=sum(r.operational_state is OperationalState.ACTIVE for r in results),
        degraded=sum(r.operational_state is OperationalState.DEGRADED for r in results),
        paused=sum(r.operational_state is OperationalState.PAUSED for r in results),
    )
