from datetime import UTC, datetime

from matamaple_trader.ai import AIReview, AIReviewDecision, review_signal
from matamaple_trader.domain import OperationalState, Signal, SignalResult


def signal_result(signal: Signal) -> SignalResult:
    return SignalResult(
        timestamp=datetime(2026, 9, 9, tzinfo=UTC),
        symbol="XAUUSD",
        signal=signal,
        raw_score=0.8,
        calibrated_probability=0.78,
        regime="TREND",
        entry_low=1.0,
        entry_high=1.1,
        stop_loss=0.9,
        tp1=1.2,
        tp2=1.3,
        tp3=1.4,
        risk_reward=2.0,
        grade="A",
        spread_state="NORMAL",
        portfolio_warning=None,
        pipeline_version="v0.2.0-fbs",
        operational_state=OperationalState.ACTIVE,
    )


def test_ai_can_confirm_actionable_signal():
    review = AIReview(AIReviewDecision.CONFIRM, 0.8, "aligned", "qwen3:4b")
    assert review_signal(signal_result(Signal.BUY), review) is Signal.BUY


def test_ai_can_only_downgrade_actionable_signal():
    review = AIReview(AIReviewDecision.REJECT, 0.9, "conflict", "qwen3:4b")
    assert review_signal(signal_result(Signal.SELL), review) is Signal.WAIT


def test_missing_ai_fails_safe_to_wait():
    assert review_signal(signal_result(Signal.BUY), None) is Signal.WAIT


def test_ai_never_upgrades_wait_or_avoid():
    confirm = AIReview(AIReviewDecision.CONFIRM, 1.0, "ok", "qwen3:4b")
    assert review_signal(signal_result(Signal.WAIT), confirm) is Signal.WAIT
    assert review_signal(signal_result(Signal.AVOID), confirm) is Signal.AVOID
