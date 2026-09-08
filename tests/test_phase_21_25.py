from datetime import UTC, datetime, timedelta

from matamaple_trader.adapters.shared import HistoricalAdapter, MockLiveAdapter
from matamaple_trader.domain import Bar, MarketSnapshot, OperationalState, PipelineContext, Signal
from matamaple_trader.grading import GradeInputs, GradeTradeResult, grade_signal, validate_grade_performance
from matamaple_trader.levels import build_price_levels
from matamaple_trader.pipeline.core import QuantSignalPipeline
from matamaple_trader.portfolio import SignalExposure, exposure_warning
from matamaple_trader.signals import SignalInputs, evaluate_signal


def bars():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(
        Bar(start + timedelta(minutes=15 * i), 100 + i * 0.1, 100.6 + i * 0.1, 99.5 + i * 0.1, 100.2 + i * 0.1)
        for i in range(12)
    )


def metadata(**overrides):
    base = {
        "candidate_signal": Signal.BUY,
        "raw_score": 1.25,
        "calibrated_probability": 0.82,
        "regime": "TREND",
        "atr": 1.0,
        "spread_points": 10.0,
        "normal_spread_points": 10.0,
        "expected_move": 3.0,
        "estimated_cost": 1.0,
        "mtf_aligned": True,
        "model_agreement": 0.9,
        "trend_alignment": True,
        "volatility_ok": True,
        "structure_score": 0.9,
    }
    base.update(overrides)
    return base


def market(**overrides):
    b = bars()
    return MarketSnapshot(b[-1].timestamp, "EURUSD", "M15", b, metadata=metadata(**overrides))


def test_signal_engine_requires_all_actionable_gates():
    decision = evaluate_signal(SignalInputs(Signal.BUY, 0.8, "TREND", 10, 10, 3, 1, 2, True))
    assert decision.signal is Signal.BUY
    low_probability = evaluate_signal(SignalInputs(Signal.BUY, 0.5, "TREND", 10, 10, 3, 1, 2, True))
    assert low_probability.signal is Signal.WAIT and "probability_gate" in low_probability.reasons


def test_high_spread_is_avoid_and_degraded_never_actionable():
    spread = evaluate_signal(SignalInputs(Signal.BUY, 0.9, "TREND", 20, 10, 3, 1, 2, True))
    assert spread.signal is Signal.AVOID
    degraded = evaluate_signal(SignalInputs(Signal.BUY, 0.9, "TREND", 10, 10, 3, 1, 2, True, OperationalState.DEGRADED))
    assert degraded.signal is Signal.WAIT


def test_levels_are_deterministic_and_not_probability_driven():
    first = build_price_levels(Signal.BUY, bars(), 1.0)
    second = build_price_levels(Signal.BUY, bars(), 1.0)
    assert first == second
    assert first.stop_loss < first.entry_low < first.entry_high < first.tp1 < first.tp2 < first.tp3
    assert first.risk_reward == 2.0


def test_grade_formula_is_transparent_and_versioned():
    result = grade_signal(GradeInputs(Signal.BUY, 0.95, True, 1.0, True, True, True, 1.0, 3.0))
    assert result.grade == "A+"
    assert result.version == "grade-v1"
    assert round(sum(result.components.values()), 6) == result.score


def test_grade_validation_detects_bad_monotonicity():
    records = [
        GradeTradeResult("A+", -1.0), GradeTradeResult("A+", -0.5),
        GradeTradeResult("A", 1.0), GradeTradeResult("A", 0.5),
    ]
    metrics, issues = validate_grade_performance(records)
    assert metrics["A+"].expectancy < metrics["A"].expectancy
    assert any("expectancy_not_monotonic" in issue for issue in issues)


def test_portfolio_warning_is_informational_only_at_three_or_more():
    exposures = [
        SignalExposure("EURUSD", Signal.BUY),
        SignalExposure("GBPUSD", Signal.BUY),
        SignalExposure("AUDUSD", Signal.BUY),
    ]
    warning = exposure_warning(exposures)
    assert warning and "3 BUY signals in USD group" in warning


def test_quant_pipeline_is_identical_across_historical_and_mock_live_adapters():
    pipeline = QuantSignalPipeline()
    context = PipelineContext("v1.0.0")
    exposures = [SignalExposure("EURUSD", Signal.BUY), SignalExposure("GBPUSD", Signal.BUY), SignalExposure("AUDUSD", Signal.BUY)]
    snapshot = market(portfolio_exposures=exposures)
    historical = HistoricalAdapter(pipeline).evaluate(snapshot, context)
    live = MockLiveAdapter(pipeline).evaluate(snapshot, context)
    assert historical == live
    assert historical.signal is Signal.BUY
    assert historical.entry_low is not None and historical.stop_loss is not None and historical.tp3 is not None
    assert historical.portfolio_warning is not None


def test_failed_gate_returns_wait_without_actionable_price_levels():
    result = QuantSignalPipeline().evaluate(market(calibrated_probability=0.40), PipelineContext("v1"))
    assert result.signal is Signal.WAIT
    assert result.entry_low is None and result.stop_loss is None and result.tp1 is None
