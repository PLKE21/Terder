from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from matamaple_trader.domain import Signal, SignalResult


class AIReviewDecision(StrEnum):
    CONFIRM = "CONFIRM"
    CAUTION = "CAUTION"
    REJECT = "REJECT"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class AIReviewInput:
    symbol: str
    quant_signal: Signal
    calibrated_probability: float | None
    regime: str
    grade: str
    spread_state: str
    risk_reward: float | None
    trend_alignment: bool
    volatility_ok: bool
    structure_score: float
    portfolio_warning: str | None = None


@dataclass(frozen=True, slots=True)
class AIReview:
    decision: AIReviewDecision
    confidence: float | None
    rationale: str
    model: str

    @property
    def allows_trade(self) -> bool:
        return self.decision is AIReviewDecision.CONFIRM


def review_signal(signal: SignalResult, ai_review: AIReview | None) -> Signal:
    """Fail-safe post-quant review gate.

    AI can only downgrade an actionable quant signal. It can never upgrade WAIT/AVOID
    into BUY/SELL and cannot bypass deterministic risk/execution controls.
    """
    if signal.signal not in {Signal.BUY, Signal.SELL}:
        return signal.signal
    if ai_review is None or ai_review.decision is AIReviewDecision.UNAVAILABLE:
        return Signal.WAIT
    if ai_review.decision is AIReviewDecision.CONFIRM:
        return signal.signal
    return Signal.WAIT
