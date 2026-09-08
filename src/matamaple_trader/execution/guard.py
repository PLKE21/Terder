from __future__ import annotations

from dataclasses import dataclass

from matamaple_trader.domain import Signal, SignalResult
from matamaple_trader.risk import RiskDecision


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    spread_points: float | None
    max_spread_points: float
    stop_distance_points: float | None
    min_stop_distance_points: float
    freeze_distance_points: float
    estimated_cost: float | None = None
    max_estimated_cost: float | None = None
    terminal_trade_allowed: bool = True
    symbol_trade_allowed: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    allowed: bool
    reason: str


class FBSExecutionGuard:
    """Final deterministic pre-trade gate for FBS MT5.

    No order is sent here. This module only decides whether an already-reviewed,
    risk-approved trade is eligible for the execution adapter.
    """

    def evaluate(self, signal: SignalResult, risk: RiskDecision, ctx: ExecutionContext) -> ExecutionDecision:
        if signal.signal not in {Signal.BUY, Signal.SELL}:
            return ExecutionDecision(False, "signal_not_actionable")
        if not risk.allowed:
            return ExecutionDecision(False, f"risk_block:{risk.reason}")
        if not ctx.terminal_trade_allowed:
            return ExecutionDecision(False, "terminal_trade_disabled")
        if not ctx.symbol_trade_allowed:
            return ExecutionDecision(False, "symbol_trade_disabled")
        if ctx.spread_points is None or ctx.spread_points > ctx.max_spread_points:
            return ExecutionDecision(False, "spread_limit")
        if ctx.stop_distance_points is None:
            return ExecutionDecision(False, "missing_stop_distance")
        required_stop = max(ctx.min_stop_distance_points, ctx.freeze_distance_points)
        if ctx.stop_distance_points < required_stop:
            return ExecutionDecision(False, "stop_or_freeze_level")
        if ctx.max_estimated_cost is not None:
            if ctx.estimated_cost is None or ctx.estimated_cost > ctx.max_estimated_cost:
                return ExecutionDecision(False, "cost_limit")
        return ExecutionDecision(True, "ok")
