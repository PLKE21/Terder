from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import numpy as np
import pandas as pd

class TaskType(StrEnum):
    REGRESSION = "REGRESSION"
    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
    MULTICLASS_CLASSIFICATION = "MULTICLASS_CLASSIFICATION"
    RANKING = "RANKING"

@dataclass(frozen=True, slots=True)
class LabelContract:
    name: str
    target_type: TaskType
    label_version: str
    mapping_rules: str
    horizon_bars: int
    thresholds: dict[str, float]

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")

    @property
    def supports_three_class_probability(self) -> bool:
        return self.target_type == TaskType.MULTICLASS_CLASSIFICATION and self.name in {"n_bar_direction", "triple_barrier"}

def forward_return_contract(horizon_bars: int = 4) -> LabelContract:
    return LabelContract("forward_return", TaskType.REGRESSION, "forward-return-v1", "(close[t+h]/close[t])-1", horizon_bars, {})

def n_bar_direction_contract(horizon_bars: int = 4, neutral_threshold: float = 0.0) -> LabelContract:
    if neutral_threshold < 0:
        raise ValueError("neutral_threshold must be >= 0")
    task = TaskType.BINARY_CLASSIFICATION if neutral_threshold == 0 else TaskType.MULTICLASS_CLASSIFICATION
    rules = "1 if forward_return>0 else 0" if task == TaskType.BINARY_CLASSIFICATION else "SELL=-1, NEUTRAL=0, BUY=1 by +/- neutral_threshold"
    return LabelContract("n_bar_direction", task, "nbar-direction-v1", rules, horizon_bars, {"neutral_threshold": neutral_threshold})

def triple_barrier_contract(horizon_bars: int = 8, upper_return: float = 0.01, lower_return: float = 0.01) -> LabelContract:
    if upper_return <= 0 or lower_return <= 0:
        raise ValueError("barriers must be positive")
    return LabelContract("triple_barrier", TaskType.MULTICLASS_CLASSIFICATION, "triple-barrier-v1", "first upper hit=BUY(1), first lower hit=SELL(-1), otherwise NEUTRAL(0)", horizon_bars, {"upper_return": upper_return, "lower_return": lower_return})

def make_labels(close: pd.Series, contract: LabelContract) -> pd.Series:
    close = close.astype(float)
    h = contract.horizon_bars
    if contract.name == "forward_return":
        return close.shift(-h) / close - 1.0
    if contract.name == "n_bar_direction":
        ret = close.shift(-h) / close - 1.0
        threshold = contract.thresholds["neutral_threshold"]
        if contract.target_type == TaskType.BINARY_CLASSIFICATION:
            return pd.Series(np.where(ret.notna(), (ret > 0).astype(int), np.nan), index=close.index, dtype=float)
        return pd.Series(np.select([ret > threshold, ret < -threshold], [1, -1], default=0), index=close.index).where(ret.notna())
    if contract.name == "triple_barrier":
        upper = contract.thresholds["upper_return"]
        lower = contract.thresholds["lower_return"]
        values: list[float] = []
        arr = close.to_numpy()
        for i, start in enumerate(arr):
            if i + h >= len(arr):
                values.append(np.nan)
                continue
            label = 0
            for future in arr[i + 1 : i + h + 1]:
                ret = future / start - 1.0
                if ret >= upper:
                    label = 1
                    break
                if ret <= -lower:
                    label = -1
                    break
            values.append(float(label))
        return pd.Series(values, index=close.index)
    raise ValueError(f"unsupported contract: {contract.name}")
