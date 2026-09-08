from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from matamaple_trader.adapters.mt5_connector import MT5Connector, MT5ConnectorError
from matamaple_trader.domain import Signal


class FakeMT5:
    COPY_TICKS_ALL = 0
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def __init__(self) -> None:
        self.started = False
        self.last_margin_action = None
        self.last_profit_action = None

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

    def order_calc_margin(self, action: int, symbol: str, volume: float, price: float):
        self.last_margin_action = action
        return 25.0

    def order_calc_profit(self, action: int, symbol: str, volume: float, price_open: float, price_close: float):
        self.last_profit_action = action
        return -10.0


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


def test_connector_supports_broker_preflight_without_order_send() -> None:
    fake = FakeMT5()
    connector = MT5Connector(fake)
    connector.connect()
    assert connector.order_calc_margin(Signal.BUY, "EURUSD", 0.1, 1.1002) == 25.0
    assert fake.last_margin_action == fake.ORDER_TYPE_BUY
    assert connector.order_calc_profit(Signal.SELL, "EURUSD", 0.1, 1.1000, 1.1010) == -10.0
    assert fake.last_profit_action == fake.ORDER_TYPE_SELL
    assert not hasattr(connector, "order_send")


def test_connector_requires_connection() -> None:
    connector = MT5Connector(FakeMT5())
    with pytest.raises(MT5ConnectorError):
        connector.get_tick("EURUSD")
    with pytest.raises(MT5ConnectorError):
        connector.order_calc_margin(Signal.BUY, "EURUSD", 0.1, 1.1)
