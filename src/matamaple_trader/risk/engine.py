from __future__ import annotations

from dataclasses import dataclass

from matamaple_trader.domain import Signal


@dataclass(frozen=True, slots=True)
class RiskConfig:
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_open_positions: int = 3
    min_free_margin_pct: float = 40.0

    def __post_init__(self) -> None:
        if not (0 < self.risk_per_trade_pct <= 5):
            raise ValueError("risk_per_trade_pct must be in (0, 5]")
        if not (0 < self.max_daily_loss_pct <= 20):
            raise ValueError("max_daily_loss_pct must be in (0, 20]")
        if not (0 < self.max_drawdown_pct <= 50):
            raise ValueError("max_drawdown_pct must be in (0, 50]")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive")
        if not (0 < self.min_free_margin_pct <= 100):
            raise ValueError("min_free_margin_pct must be in (0, 100]")


@dataclass(frozen=True, slots=True)
class AccountRiskState:
    equity: float
    balance: float
    daily_pnl: float
    peak_equity: float
    free_margin: float
    open_positions: int = 0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    reason: str
    max_risk_amount: float


class RiskEngine:
    """Deterministic account-level risk gate.

    This engine never creates trading signals. It only accepts/rejects an already
    reviewed BUY/SELL candidate and computes the maximum risk budget.
    """

    def __init__(self, config: RiskConfig = RiskConfig()) -> None:
        self.config = config

    def evaluate(self, signal: Signal, state: AccountRiskState) -> RiskDecision:
        if signal not in {Signal.BUY, Signal.SELL}:
            return RiskDecision(False, "signal_not_actionable", 0.0)
        if state.equity <= 0 or state.balance <= 0 or state.peak_equity <= 0:
            return RiskDecision(False, "invalid_account_state", 0.0)
        if state.open_positions >= self.config.max_open_positions:
            return RiskDecision(False, "max_open_positions", 0.0)

        daily_loss_pct = max(0.0, -state.daily_pnl / state.balance * 100.0)
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            return RiskDecision(False, "daily_loss_limit", 0.0)

        drawdown_pct = max(0.0, (state.peak_equity - state.equity) / state.peak_equity * 100.0)
        if drawdown_pct >= self.config.max_drawdown_pct:
            return RiskDecision(False, "max_drawdown_limit", 0.0)

        free_margin_pct = state.free_margin / state.equity * 100.0
        if free_margin_pct < self.config.min_free_margin_pct:
            return RiskDecision(False, "insufficient_free_margin", 0.0)

        amount = state.equity * self.config.risk_per_trade_pct / 100.0
        return RiskDecision(True, "ok", float(amount))
