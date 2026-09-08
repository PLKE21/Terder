from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DemoFillComparison:
    symbol: str
    side: str
    expected_fill: float
    actual_fill: float
    point: float
    slippage_points: float
    expected_spread_points: float | None = None
    actual_spread_points: float | None = None


class DemoValidator:
    """XM demo validation for manually executed trades only. No order placement API exists here."""

    @staticmethod
    def compare_fill(
        *,
        symbol: str,
        side: str,
        expected_fill: float,
        actual_fill: float,
        point: float,
        expected_spread_points: float | None = None,
        actual_spread_points: float | None = None,
    ) -> DemoFillComparison:
        if point <= 0:
            raise ValueError("point must be positive")
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        adverse = actual_fill - expected_fill if normalized_side == "BUY" else expected_fill - actual_fill
        return DemoFillComparison(
            symbol=symbol,
            side=normalized_side,
            expected_fill=float(expected_fill),
            actual_fill=float(actual_fill),
            point=float(point),
            slippage_points=float(adverse / point),
            expected_spread_points=expected_spread_points,
            actual_spread_points=actual_spread_points,
        )

    @staticmethod
    def summarize(comparisons: list[DemoFillComparison] | tuple[DemoFillComparison, ...]) -> dict[str, float | int]:
        if not comparisons:
            return {"count": 0, "mean_slippage_points": 0.0, "mean_abs_slippage_points": 0.0}
        values = [c.slippage_points for c in comparisons]
        return {
            "count": len(values),
            "mean_slippage_points": float(sum(values) / len(values)),
            "mean_abs_slippage_points": float(sum(abs(v) for v in values) / len(values)),
        }
