from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import floor
from typing import Protocol

from matamaple_trader.domain import Signal
from matamaple_trader.execution.guard import ExecutionDecision
from matamaple_trader.risk import RiskDecision


class ExecutionMode(StrEnum):
    DRY_RUN = "DRY_RUN"
    DEMO = "DEMO"
    LIVE = "LIVE"


@dataclass(frozen=True, slots=True)
class OrderPlan:
    symbol: str
    side: Signal
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float | None
    deviation_points: int = 10


@dataclass(frozen=True, slots=True)
class PreflightResult:
    allowed: bool
    reason: str
    estimated_margin: float | None = None
    estimated_loss_at_stop: float | None = None


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    mode: ExecutionMode
    submitted: bool
    reason: str
    plan: OrderPlan


class MT5Preflight(Protocol):
    def order_calc_margin(self, side: Signal, symbol: str, volume: float, price: float) -> float | None: ...
    def order_calc_profit(self, side: Signal, symbol: str, volume: float, open_price: float, close_price: float) -> float | None: ...


def normalize_volume(raw_volume: float, *, volume_min: float, volume_max: float, volume_step: float) -> float:
    if raw_volume <= 0 or volume_min <= 0 or volume_max < volume_min or volume_step <= 0:
        raise ValueError("invalid volume inputs")
    capped = min(raw_volume, volume_max)
    steps = floor((capped - volume_min) / volume_step + 1e-12)
    normalized = volume_min + max(0, steps) * volume_step
    if raw_volume < volume_min:
        return 0.0
    return round(min(normalized, volume_max), 8)


def size_volume_from_stop(
    *,
    max_risk_amount: float,
    entry_price: float,
    stop_loss: float,
    tick_size: float,
    tick_value: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> float:
    distance = abs(entry_price - stop_loss)
    if max_risk_amount <= 0 or distance <= 0 or tick_size <= 0 or tick_value <= 0:
        return 0.0
    loss_per_lot = distance / tick_size * tick_value
    raw = max_risk_amount / loss_per_lot
    return normalize_volume(raw, volume_min=volume_min, volume_max=volume_max, volume_step=volume_step)


class FBSOrderAdapter:
    """Execution boundary with DRY_RUN as the safe default.

    This version intentionally refuses DEMO/LIVE submission. It performs deterministic
    preflight calculations only. A later audited adapter may implement order_send.
    """

    def __init__(self, preflight: MT5Preflight, mode: ExecutionMode = ExecutionMode.DRY_RUN) -> None:
        self.preflight = preflight
        self.mode = mode

    def preflight_order(self, plan: OrderPlan, risk: RiskDecision, guard: ExecutionDecision, *, free_margin: float) -> PreflightResult:
        if not guard.allowed:
            return PreflightResult(False, f"execution_guard:{guard.reason}")
        if not risk.allowed:
            return PreflightResult(False, f"risk:{risk.reason}")
        if plan.side not in {Signal.BUY, Signal.SELL} or plan.volume <= 0:
            return PreflightResult(False, "invalid_order_plan")
        if plan.deviation_points < 0:
            return PreflightResult(False, "invalid_deviation")

        margin = self.preflight.order_calc_margin(plan.side, plan.symbol, plan.volume, plan.entry_price)
        if margin is None or margin <= 0:
            return PreflightResult(False, "margin_calculation_failed")
        if margin > free_margin:
            return PreflightResult(False, "insufficient_margin", float(margin))

        pnl = self.preflight.order_calc_profit(plan.side, plan.symbol, plan.volume, plan.entry_price, plan.stop_loss)
        if pnl is None:
            return PreflightResult(False, "profit_calculation_failed", float(margin))
        estimated_loss = max(0.0, -float(pnl))
        if estimated_loss > risk.max_risk_amount * 1.001:
            return PreflightResult(False, "risk_budget_exceeded", float(margin), estimated_loss)
        return PreflightResult(True, "ok", float(margin), estimated_loss)

    def execute(self, plan: OrderPlan, preflight: PreflightResult) -> ExecutionReceipt:
        if not preflight.allowed:
            return ExecutionReceipt(self.mode, False, f"preflight:{preflight.reason}", plan)
        if self.mode is not ExecutionMode.DRY_RUN:
            return ExecutionReceipt(self.mode, False, "order_send_disabled_until_demo_audit", plan)
        return ExecutionReceipt(self.mode, False, "dry_run_validated_no_order_sent", plan)
