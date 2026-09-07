from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

REQUIRED_OHLC = ("timestamp", "open", "high", "low", "close")

@dataclass(frozen=True, slots=True)
class FeatureConfig:
    atr_window: int = 14
    rsi_window: int = 14
    fast_ema: int = 12
    slow_ema: int = 26
    volatility_window: int = 20
    feature_version: str = "features-v1"

class FeatureEngine:
    """Causal feature engine for completed candles only."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def transform(self, bars: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_OHLC if c not in bars.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        frame = bars.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if not frame["timestamp"].is_monotonic_increasing:
            raise ValueError("bars must be chronological")
        if frame["timestamp"].duplicated().any():
            raise ValueError("duplicate timestamps are not allowed")

        prev_close = frame["close"].shift(1)
        frame["return_1"] = frame["close"].pct_change(fill_method=None)
        frame["log_return_1"] = np.log(frame["close"] / prev_close)
        frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
        frame["body_pct"] = (frame["close"] - frame["open"]) / frame["open"].replace(0, np.nan)

        tr = pd.concat([
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        frame["atr"] = tr.rolling(self.config.atr_window, min_periods=self.config.atr_window).mean()
        frame["atr_pct"] = frame["atr"] / frame["close"].replace(0, np.nan)

        delta = frame["close"].diff()
        gain = delta.clip(lower=0).rolling(self.config.rsi_window, min_periods=self.config.rsi_window).mean()
        loss = (-delta.clip(upper=0)).rolling(self.config.rsi_window, min_periods=self.config.rsi_window).mean()
        rs = gain / loss.replace(0, np.nan)
        frame["rsi"] = 100 - (100 / (1 + rs))

        frame["ema_fast"] = frame["close"].ewm(span=self.config.fast_ema, adjust=False).mean()
        frame["ema_slow"] = frame["close"].ewm(span=self.config.slow_ema, adjust=False).mean()
        frame["ema_spread_pct"] = (frame["ema_fast"] - frame["ema_slow"]) / frame["close"].replace(0, np.nan)
        frame["volatility"] = frame["return_1"].rolling(self.config.volatility_window, min_periods=self.config.volatility_window).std()
        frame["utc_hour"] = frame["timestamp"].dt.hour.astype("int16")
        frame["utc_dayofweek"] = frame["timestamp"].dt.dayofweek.astype("int8")
        return frame


def align_closed_higher_timeframe(
    base: pd.DataFrame,
    higher: pd.DataFrame,
    *,
    higher_bar_seconds: int,
    suffix: str,
) -> pd.DataFrame:
    """Backward/as-of align HTF values only after the higher-timeframe bar has closed."""
    if higher_bar_seconds <= 0:
        raise ValueError("higher_bar_seconds must be positive")
    left = base.copy()
    right = higher.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
    if not left["timestamp"].is_monotonic_increasing or not right["timestamp"].is_monotonic_increasing:
        raise ValueError("inputs must be chronological")
    right["available_at"] = right["timestamp"] + pd.to_timedelta(higher_bar_seconds, unit="s")
    payload = [c for c in right.columns if c not in {"timestamp", "available_at"}]
    right = right[["available_at", *payload]].rename(columns={c: f"{c}_{suffix}" for c in payload})
    return pd.merge_asof(left, right, left_on="timestamp", right_on="available_at", direction="backward")
