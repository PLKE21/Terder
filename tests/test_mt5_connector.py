from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from matamaple_trader.adapters.mt5_connector import MT5Connector, MT5ConnectorError


class FakeMT5:
    COPY_TICKS_ALL = 0

    def __init__(self) -> None:
        self.started = False

    def initialize(self, **kwargs: object) -> bool:
        self.started = True
        return True

    def shutdown(self) -> None:
        self.started = False

    def last_error(self) -> tuple[int, str]:
        return (0, "ok")

    def symbol_info_tick(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(time_msc=1_700_000_000_123, bid=1.1000, ask=1.1002, last=0.0)

    def symbol_info(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            trade_contract_size=100_000.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            swap_long=-1.1,
            swap_short=0.4,
        )

    def copy_rates_range(self, symbol: str, timeframe: int, start: datetime, end: datetime):
        return [
            {"time": 1_700_000_000, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "tick_volume": 10}
        ]


def test_connector_is_read_only_and_returns_utc_data() -> None:
    fake = FakeMT5()
    connector = MT5Connector(fake)
    connector.connect()
    tick = connector.get_tick("EURUSD")
    assert tick.timestamp.tzinfo is UTC
    assert tick.spread == pytest.approx(0.0002)
    assert not hasattr(connector, "order_send")
    frame = connector.get_rates(
        "EURUSD", 15, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 1, 2, tzinfo=UTC)
    )
    assert isinstance(frame, pd.DataFrame)
    assert str(frame["timestamp"].dt.tz) == "UTC"
    connector.close()


def test_connector_requires_connection() -> None:
    with pytest.raises(MT5ConnectorError):
        MT5Connector(FakeMT5()).get_tick("EURUSD")
