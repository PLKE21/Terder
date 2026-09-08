from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np


@dataclass(frozen=True, slots=True)
class PurgedSplitConfig:
    n_splits: int = 5
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.embargo_bars < 0:
            raise ValueError("embargo_bars must be >= 0")


def _validate_end_times(n: int, event_end_index) -> np.ndarray:
    if event_end_index is None:
        return np.arange(n, dtype=int)
    out = np.asarray(event_end_index, dtype=int)
    if out.shape != (n,):
        raise ValueError("event_end_index must have one value per sample")
    if np.any(out < np.arange(n)):
        raise ValueError("event_end_index cannot precede sample index")
    return out


class PurgedKFold:
    """Chronological K-fold with event-overlap purging and post-test embargo."""

    def __init__(self, config: PurgedSplitConfig = PurgedSplitConfig()) -> None:
        self.config = config

    def split(self, X, *, event_end_index=None):
        n = len(X)
        if n < self.config.n_splits:
            raise ValueError("not enough samples for requested n_splits")
        ends = _validate_end_times(n, event_end_index)
        indices = np.arange(n, dtype=int)
        folds = np.array_split(indices, self.config.n_splits)
        for test in folds:
            test_start, test_end = int(test[0]), int(test[-1])
            overlap = (indices <= test_end) & (ends >= test_start)
            embargo = (indices > test_end) & (indices <= test_end + self.config.embargo_bars)
            train = indices[~(overlap | embargo)]
            yield train, test


class CombinatorialPurgedCV:
    """CPCV research splitter using combinations of chronological groups as test sets."""

    def __init__(self, *, n_groups: int = 6, test_groups: int = 2, embargo_bars: int = 0) -> None:
        if n_groups < 3:
            raise ValueError("n_groups must be >= 3")
        if test_groups <= 0 or test_groups >= n_groups:
            raise ValueError("test_groups must be between 1 and n_groups-1")
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be >= 0")
        self.n_groups = n_groups
        self.test_groups = test_groups
        self.embargo_bars = embargo_bars

    def split(self, X, *, event_end_index=None):
        n = len(X)
        if n < self.n_groups:
            raise ValueError("not enough samples for requested n_groups")
        ends = _validate_end_times(n, event_end_index)
        indices = np.arange(n, dtype=int)
        groups = np.array_split(indices, self.n_groups)
        for chosen in combinations(range(self.n_groups), self.test_groups):
            test = np.concatenate([groups[i] for i in chosen])
            forbidden = np.zeros(n, dtype=bool)
            for i in chosen:
                group = groups[i]
                start, end = int(group[0]), int(group[-1])
                forbidden |= (indices <= end) & (ends >= start)
                forbidden |= (indices > end) & (indices <= end + self.embargo_bars)
            train = indices[~forbidden]
            yield train, np.sort(test)
