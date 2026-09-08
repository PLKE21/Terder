from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from matamaple_trader.domain import Bar, Signal


@dataclass(frozen=True, slots=True)
class LevelConfig:
    swing_lookback: int = 10
    atr_stop_multiple: float = 1.5
    entry_zone_atr_fraction: float = 0.10
    tp_r_multiples: tuple[float, float, float] = (1.0, 2.0, 3.0)


@dataclass(frozen=True, slots=True)
class PriceLevels:
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk_reward: float


def build_price_levels(candidate: Signal, bars: Sequence[Bar], atr: float, config: LevelConfig = LevelConfig()) -> PriceLevels:
    if candidate not in {Signal.BUY, Signal.SELL}:
        raise ValueError("price levels require BUY or SELL candidate")
    if not bars:
        raise ValueError("bars are required")
    if atr <= 0:
        raise ValueError("atr must be positive")

    recent = tuple(bars[-max(1, config.swing_lookback):])
    center = float(recent[-1].close)
    zone = atr * config.entry_zone_atr_fraction
    entry_low, entry_high = center - zone, center + zone
    entry_mid = (entry_low + entry_high) / 2.0

    if candidate is Signal.BUY:
        structural_stop = min(float(bar.low) for bar in recent)
        stop_loss = min(structural_stop, center - atr * config.atr_stop_multiple)
        risk = entry_mid - stop_loss
        if risk <= 0:
            raise ValueError("invalid BUY risk distance")
        targets = tuple(entry_mid + risk * multiple for multiple in config.tp_r_multiples)
    else:
        structural_stop = max(float(bar.high) for bar in recent)
        stop_loss = max(structural_stop, center + atr * config.atr_stop_multiple)
        risk = stop_loss - entry_mid
        if risk <= 0:
            raise ValueError("invalid SELL risk distance")
        targets = tuple(entry_mid - risk * multiple for multiple in config.tp_r_multiples)

    risk_reward = abs(targets[1] - entry_mid) / risk
    return PriceLevels(entry_low, entry_high, stop_loss, targets[0], targets[1], targets[2], risk_reward)
