from __future__ import annotations

from dataclasses import dataclass
from math import inf

GRADE_ORDER = ("A+", "A", "A-", "B", "C")


@dataclass(frozen=True, slots=True)
class GradeTradeResult:
    grade: str
    r_multiple: float


@dataclass(frozen=True, slots=True)
class GradeMetrics:
    grade: str
    count: int
    win_rate: float
    expectancy: float
    profit_factor: float
    max_drawdown_r: float
    average_r: float


def _metrics(grade: str, values: list[float]) -> GradeMetrics:
    count = len(values)
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    win_rate = len(wins) / count if count else 0.0
    expectancy = sum(values) / count if count else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = inf if gross_profit > 0 and gross_loss == 0 else (gross_profit / gross_loss if gross_loss else 0.0)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return GradeMetrics(grade, count, win_rate, expectancy, profit_factor, max_drawdown, expectancy)


def validate_grade_performance(records: list[GradeTradeResult], *, tolerance: float = 0.0) -> tuple[dict[str, GradeMetrics], tuple[str, ...]]:
    grouped: dict[str, list[float]] = {grade: [] for grade in GRADE_ORDER}
    for record in records:
        if record.grade in grouped:
            grouped[record.grade].append(float(record.r_multiple))
    metrics = {grade: _metrics(grade, values) for grade, values in grouped.items() if values}

    issues: list[str] = []
    present = [grade for grade in GRADE_ORDER if grade in metrics]
    for higher, lower in zip(present, present[1:]):
        high = metrics[higher]
        low = metrics[lower]
        if high.expectancy + tolerance < low.expectancy:
            issues.append(f"expectancy_not_monotonic:{higher}<{lower}")
        if high.win_rate + tolerance < low.win_rate:
            issues.append(f"win_rate_not_monotonic:{higher}<{lower}")
    return metrics, tuple(issues)
