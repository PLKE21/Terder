from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    min_train_bars: int
    test_bars: int
    step_bars: int | None = None
    gap_bars: int = 0

    def __post_init__(self) -> None:
        if self.min_train_bars <= 0 or self.test_bars <= 0:
            raise ValueError("min_train_bars and test_bars must be positive")
        if self.step_bars is not None and self.step_bars <= 0:
            raise ValueError("step_bars must be positive")
        if self.gap_bars < 0:
            raise ValueError("gap_bars must be >= 0")


class ExpandingWalkForward:
    """Expanding-window train sets with strictly later test windows."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def split(self, X):
        n = len(X)
        step = self.config.step_bars or self.config.test_bars
        train_end = self.config.min_train_bars
        while True:
            test_start = train_end + self.config.gap_bars
            test_end = test_start + self.config.test_bars
            if test_end > n:
                break
            train = np.arange(0, train_end, dtype=int)
            test = np.arange(test_start, test_end, dtype=int)
            yield train, test
            train_end += step
