from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import numpy as np


class MarketRegime(StrEnum):
    TREND = "TREND"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    trend_abs_threshold: float
    high_vol_threshold: float
    low_vol_threshold: float
    version: str = "regime-v1"

    def __post_init__(self) -> None:
        if self.trend_abs_threshold < 0:
            raise ValueError("trend_abs_threshold must be >= 0")
        if self.low_vol_threshold > self.high_vol_threshold:
            raise ValueError("low_vol_threshold must be <= high_vol_threshold")


class RegimeEngine:
    """Thresholds are fit only on development/validation data, never frozen holdout."""

    def __init__(self, thresholds: RegimeThresholds | None = None) -> None:
        self.thresholds = thresholds

    def fit(self, trend_strength, volatility, *, source: str) -> "RegimeEngine":
        source_norm = source.strip().lower()
        if source_norm in {"holdout", "frozen_holdout", "research_holdout", "final_holdout"}:
            raise RuntimeError("regime thresholds must not be tuned on frozen holdout")
        if source_norm not in {"train", "training", "validation", "development"}:
            raise ValueError("source must be training/development/validation")
        trend = np.asarray(trend_strength, dtype=float)
        vol = np.asarray(volatility, dtype=float)
        if trend.size == 0 or vol.size == 0 or trend.size != vol.size:
            raise ValueError("trend_strength and volatility must be non-empty and aligned")
        self.thresholds = RegimeThresholds(
            trend_abs_threshold=float(np.nanquantile(np.abs(trend), 0.65)),
            high_vol_threshold=float(np.nanquantile(vol, 0.75)),
            low_vol_threshold=float(np.nanquantile(vol, 0.25)),
        )
        return self

    def classify(self, *, trend_strength: float, volatility: float) -> MarketRegime:
        if self.thresholds is None:
            raise RuntimeError("regime engine is not fitted")
        t = self.thresholds
        if volatility >= t.high_vol_threshold:
            return MarketRegime.HIGH_VOLATILITY
        if volatility <= t.low_vol_threshold:
            return MarketRegime.LOW_VOLATILITY
        if abs(trend_strength) >= t.trend_abs_threshold:
            return MarketRegime.TREND
        return MarketRegime.RANGE
