from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, Sequence

import pandas as pd

from matamaple_trader.domain import SymbolSpec, Tick


class MT5Module(Protocol):
    COPY_TICKS_ALL: int

    def initialize(self, *args: Any, **kwargs: Any) -> bool: ...
    def shutdown(self) -> None: ...
    def last_error(self) -> Any: ...
    def symbol_info(self, symbol: str) -> Any: ...
    def symbol_info_tick(self, symbol: str) -> Any: ...
    def copy_rates_range(self, symbol: str, timeframe: int, date_from: datetime, date_to: datetime) -> Any: ...


class MT5ConnectorError(RuntimeError):
    pass


class MT5Connector:
    """Read-only MT5 adapter.

    This adapter intentionally exposes market-data and symbol-spec methods only.
    It has no trade execution API.
    """

    def __init__(self, mt5_module: MT5Module | None = None) -> None:
        if mt5_module is None:
            try:
                import MetaTrader5 as mt5_module  # type: ignore[no-redef]
            except ImportError as exc:
                raise MT5ConnectorError(
                    "MetaTrader5 package is not installed. Install the optional 'mt5' dependency on Windows."
                ) from exc
        self._mt5 = mt5_module
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, **initialize_kwargs: Any) -> None:
        if not self._mt5.initialize(**initialize_kwargs):
            raise MT5ConnectorError(f"MT5 initialize failed: {self._mt5.last_error()!r}")
        self._connected = True

    def close(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise MT5ConnectorError("MT5 connector is not connected")

    def get_tick(self, symbol: str) -> Tick:
        self._require_connection()
        raw = self._mt5.symbol_info_tick(symbol)
        if raw is None:
            raise MT5ConnectorError(f"No tick available for {symbol}")
        timestamp = datetime.fromtimestamp(float(raw.time_msc) / 1000.0, tz=UTC)
        last = float(raw.last) if getattr(raw, "last", 0.0) else None
        return Tick(timestamp=timestamp, bid=float(raw.bid), ask=float(raw.ask), last=last)

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        self._require_connection()
        raw = self._mt5.symbol_info(symbol)
        if raw is None:
            raise MT5ConnectorError(f"No symbol info available for {symbol}")
        return SymbolSpec(
            symbol=symbol,
            digits=int(raw.digits),
            point=float(raw.point),
            tick_size=float(raw.trade_tick_size),
            tick_value=float(raw.trade_tick_value),
            contract_size=float(raw.trade_contract_size),
            volume_min=float(raw.volume_min),
            volume_max=float(raw.volume_max),
            volume_step=float(raw.volume_step),
            swap_long=float(raw.swap_long),
            swap_short=float(raw.swap_short),
        )

    def get_rates(
        self,
        symbol: str,
        timeframe: int,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        self._require_connection()
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("start_utc and end_utc must be timezone-aware")
        start_utc = start_utc.astimezone(UTC)
        end_utc = end_utc.astimezone(UTC)
        rates: Sequence[Any] | None = self._mt5.copy_rates_range(symbol, timeframe, start_utc, end_utc)
        if rates is None:
            raise MT5ConnectorError(f"copy_rates_range failed: {self._mt5.last_error()!r}")
        frame = pd.DataFrame(rates)
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        return frame.drop(columns=["time"])
