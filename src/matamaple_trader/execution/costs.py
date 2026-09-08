from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class CostConfig:
    slippage_points: float = 0.0
    commission_per_lot: float = 0.0
    point_value_per_lot: float = 1.0

    def __post_init__(self) -> None:
        if self.slippage_points < 0 or self.commission_per_lot < 0 or self.point_value_per_lot <= 0:
            raise ValueError("cost configuration values are invalid")


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    spread_cost: float
    slippage: float
    gap_slippage: float
    swap: float
    commission: float

    @property
    def total(self) -> float:
        return self.spread_cost + self.slippage + self.gap_slippage + self.swap + self.commission


@dataclass(frozen=True, slots=True)
class FillResult:
    requested_price: float
    simulated_fill: float
    costs: CostBreakdown


class CostEngine:
    def __init__(self, config: CostConfig = CostConfig()) -> None:
        self.config = config

    def simulate_entry(self, *, side: Side | str, requested_price: float, bid: float, ask: float, point: float, lots: float = 1.0) -> FillResult:
        side = Side(side)
        if point <= 0 or lots <= 0 or ask < bid:
            raise ValueError("invalid market/fill inputs")
        market = ask if side == Side.BUY else bid
        slip_price = self.config.slippage_points * point
        fill = market + slip_price if side == Side.BUY else market - slip_price
        spread_points = (ask - bid) / point
        spread_cost = spread_points * self.config.point_value_per_lot * lots
        slippage_cost = self.config.slippage_points * self.config.point_value_per_lot * lots
        costs = CostBreakdown(spread_cost, slippage_cost, 0.0, 0.0, self.config.commission_per_lot * lots)
        return FillResult(float(requested_price), float(fill), costs)

    def simulate_stop_fill(self, *, side: Side | str, stop_price: float, next_bid: float, next_ask: float, point: float, lots: float = 1.0) -> FillResult:
        """If price gaps through SL, fill at the first realistically executable quote, not the requested stop."""
        side = Side(side)
        if point <= 0 or lots <= 0 or next_ask < next_bid:
            raise ValueError("invalid market/fill inputs")
        executable = next_bid if side == Side.BUY else next_ask
        if side == Side.BUY:
            fill = min(stop_price, executable)
            adverse_gap = max(0.0, stop_price - fill)
        else:
            fill = max(stop_price, executable)
            adverse_gap = max(0.0, fill - stop_price)
        gap_points = adverse_gap / point
        costs = CostBreakdown(0.0, 0.0, gap_points * self.config.point_value_per_lot * lots, 0.0, 0.0)
        return FillResult(float(stop_price), float(fill), costs)

    def overnight_cost(self, *, swap_points: float, nights: int, lots: float = 1.0) -> float:
        if nights < 0 or lots <= 0:
            raise ValueError("nights/lots are invalid")
        return float(swap_points * nights * self.config.point_value_per_lot * lots)

    def full_costs(self, *, entry: FillResult, swap_cost: float = 0.0) -> CostBreakdown:
        c = entry.costs
        return CostBreakdown(c.spread_cost, c.slippage, c.gap_slippage, float(swap_cost), c.commission)
