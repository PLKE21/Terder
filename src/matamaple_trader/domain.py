from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Signal(StrEnum):
    BUY = "BUY"
    WAIT = "WAIT"
    SELL = "SELL"
    AVOID = "AVOID"


class OperationalState(StrEnum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"


@dataclass(frozen=True, slots=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int = 0
    spread_points: int = 0


@dataclass(frozen=True, slots=True)
class Tick:
    timestamp: datetime
    bid: float
    ask: float
    last: float | None = None

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    symbol: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    swap_long: float
    swap_short: float
