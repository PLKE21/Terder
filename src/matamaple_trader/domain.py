from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

class Signal(StrEnum): BUY='BUY'; WAIT='WAIT'; SELL='SELL'; AVOID='AVOID'
class OperationalState(StrEnum): ACTIVE='ACTIVE'; DEGRADED='DEGRADED'; PAUSED='PAUSED'
@dataclass(frozen=True, slots=True)
class Bar:
    timestamp: datetime; open: float; high: float; low: float; close: float; tick_volume: int=0; spread_points: int=0
@dataclass(frozen=True, slots=True)
class Tick:
    timestamp: datetime; bid: float; ask: float; last: float|None=None
    @property
    def spread(self)->float: return self.ask-self.bid
@dataclass(frozen=True, slots=True)
class SymbolSpec:
    symbol: str; digits:int; point:float; tick_size:float; tick_value:float; contract_size:float; volume_min:float; volume_max:float; volume_step:float; swap_long:float; swap_short:float
@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    timestamp: datetime; symbol: str; timeframe: str; bars: tuple[Bar,...]; tick: Tick|None=None; metadata: dict[str, object]=field(default_factory=dict)
@dataclass(frozen=True, slots=True)
class PipelineContext:
    pipeline_version: str; operational_state: OperationalState=OperationalState.ACTIVE; source: str='unknown'
@dataclass(frozen=True, slots=True)
class SignalResult:
    timestamp: datetime; symbol: str; signal: Signal; raw_score: float|None; calibrated_probability: float|None; regime: str; entry_low: float|None; entry_high: float|None; stop_loss: float|None; tp1: float|None; tp2: float|None; tp3: float|None; risk_reward: float|None; grade: str; spread_state: str; portfolio_warning: str|None; pipeline_version: str; operational_state: OperationalState
