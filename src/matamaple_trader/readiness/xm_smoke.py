from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd

from matamaple_trader.broker.spec_integrity import BrokerSpecMonitor
from matamaple_trader.data.integrity import validate_bars
from matamaple_trader.domain import OperationalState


@dataclass(frozen=True, slots=True)
class XMReadinessReport:
    symbol: str
    checked_at: datetime
    tick_age_seconds: float
    bar_count: int
    quality_failures: tuple[str,...]
    broker_state: OperationalState
    ready_for_collection: bool
    continuity_checked: bool


def run_readiness_check(
    connector,
    *,
    symbol: str,
    timeframe_code: int,
    start_utc: datetime,
    end_utc: datetime,
    expected_delta: pd.Timedelta | None,
    min_bars: int = 100,
    max_spread_points: int | None = None,
    max_tick_age_seconds: float = 10.0,
    broker_history_path: str | Path | None = None,
    now_utc: datetime | None = None,
) -> XMReadinessReport:
    """Read-only XM/MT5 smoke check.

    The connector is expected to expose only get_tick/get_symbol_spec/get_rates.
    This function never places orders and has no execution API.
    """
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError('start_utc/end_utc must be timezone-aware')
    if end_utc <= start_utc:
        raise ValueError('end_utc must be after start_utc')
    if max_tick_age_seconds <= 0:
        raise ValueError('max_tick_age_seconds must be positive')

    checked_at=(now_utc or datetime.now(UTC)).astimezone(UTC)
    tick=connector.get_tick(symbol)
    spec=connector.get_symbol_spec(symbol)
    frame=connector.get_rates(symbol,timeframe_code,start_utc.astimezone(UTC),end_utc.astimezone(UTC)).copy()
    if not frame.empty:
        frame['timestamp']=pd.to_datetime(frame['timestamp'],utc=True)

    monitor=BrokerSpecMonitor(broker_history_path)
    monitor.observe(spec,checked_at)
    tick_age=max(0.0,(checked_at-tick.timestamp.astimezone(UTC)).total_seconds())

    report=validate_bars(
        frame,
        expected_delta,
        min_bars=min_bars,
        max_spread_points=max_spread_points,
        latest_tick_timestamp=tick.timestamp,
        reference_timestamp=checked_at,
        max_tick_age=pd.Timedelta(seconds=max_tick_age_seconds),
    )
    state=monitor.state(symbol)
    ready=report.ok and state is OperationalState.ACTIVE
    return XMReadinessReport(
        symbol=symbol,
        checked_at=checked_at,
        tick_age_seconds=tick_age,
        bar_count=len(frame),
        quality_failures=report.failures,
        broker_state=state,
        ready_for_collection=ready,
        continuity_checked=expected_delta is not None,
    )
