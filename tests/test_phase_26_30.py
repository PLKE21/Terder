from datetime import UTC, datetime, timedelta

import pytest

from matamaple_trader.backtest import BacktestEngine
from matamaple_trader.dashboard import build_dashboard_snapshot
from matamaple_trader.demo import DemoValidator
from matamaple_trader.domain import Bar, MarketSnapshot, OperationalState, PipelineContext, Signal
from matamaple_trader.drift import DriftConfig, DriftMonitor, DriftObservation
from matamaple_trader.pipeline.core import QuantSignalPipeline
from matamaple_trader.shadow import ShadowRunner, ShadowStore


def bars(offset=0):
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * offset)
    return tuple(
        Bar(start + timedelta(minutes=15 * i), 100 + i * 0.1, 100.6 + i * 0.1, 99.5 + i * 0.1, 100.2 + i * 0.1)
        for i in range(12)
    )


def market(offset=0, **overrides):
    b = bars(offset)
    meta = {
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
    meta.update(overrides)
    return MarketSnapshot(b[-1].timestamp, "EURUSD", "M15", b, metadata=meta)


def test_backtest_replays_shared_pipeline_without_strategy_reimplementation():
    pipeline = QuantSignalPipeline()
    report = BacktestEngine(pipeline).run([market(0), market(20)], PipelineContext("v1"))
    assert len(report.records) == 2
    assert report.actionable_count == 2
    assert all(r.signal is Signal.BUY for r in report.records)
    assert all(r.execution_timestamp >= r.signal_timestamp for r in report.records)


def test_backtest_rejects_non_chronological_replay():
    with pytest.raises(ValueError, match="chronological"):
        BacktestEngine(QuantSignalPipeline()).run([market(20), market(0)], PipelineContext("v1"))


def test_shadow_predictions_are_immutable_and_outcomes_are_separate(tmp_path):
    store = ShadowStore(tmp_path / "shadow.sqlite")
    runner = ShadowRunner(QuantSignalPipeline(), store)
    result, prediction_id = runner.evaluate_and_store(market(), PipelineContext("v1"))
    assert result.signal is Signal.BUY
    store.attach_outcome(prediction_id, {"realized_r": 1.5})
    with pytest.raises(RuntimeError):
        store.attach_outcome(prediction_id, {"realized_r": -1.0})
    with pytest.raises(RuntimeError):
        runner.evaluate_and_store(market(), PipelineContext("v1"))
    assert len(store.list_predictions()) == 1


def test_drift_hysteresis_and_critical_broker_bypass():
    monitor = DriftMonitor(DriftConfig(degrade_persistence=3, pause_persistence=2, recovery_persistence=2))
    breach = DriftObservation(feature_drift=True)
    assert monitor.observe("EURUSD:model", breach) is OperationalState.ACTIVE
    assert monitor.observe("EURUSD:model", breach) is OperationalState.ACTIVE
    assert monitor.observe("EURUSD:model", breach) is OperationalState.DEGRADED

    critical = DriftMonitor()
    assert critical.observe("EURUSD:model", DriftObservation(critical_broker_spec=True)) is OperationalState.DEGRADED


def test_drift_paused_recovers_in_stages():
    monitor = DriftMonitor(DriftConfig(degrade_persistence=1, pause_persistence=2, recovery_persistence=2))
    severe = DriftObservation(performance_drift=True, severe=True)
    assert monitor.observe("x", severe) is OperationalState.DEGRADED
    assert monitor.observe("x", severe) is OperationalState.PAUSED
    clean = DriftObservation()
    assert monitor.observe("x", clean) is OperationalState.PAUSED
    assert monitor.observe("x", clean) is OperationalState.DEGRADED
    assert monitor.observe("x", clean) is OperationalState.DEGRADED
    assert monitor.observe("x", clean) is OperationalState.ACTIVE


def test_dashboard_view_model_preserves_operational_state_and_pipeline_version():
    pipeline = QuantSignalPipeline()
    active = pipeline.evaluate(market(), PipelineContext("v1", OperationalState.ACTIVE))
    degraded = pipeline.evaluate(market(20), PipelineContext("v1", OperationalState.DEGRADED))
    snapshot = build_dashboard_snapshot([active, degraded])
    assert snapshot.active == 1 and snapshot.degraded == 1 and snapshot.paused == 0
    assert {row.pipeline_version for row in snapshot.rows} == {"v1"}


def test_demo_validator_only_compares_manual_fills():
    comparison = DemoValidator.compare_fill(
        symbol="EURUSD",
        side="BUY",
        expected_fill=1.10000,
        actual_fill=1.10003,
        point=0.00001,
        expected_spread_points=10,
        actual_spread_points=12,
    )
    assert round(comparison.slippage_points, 6) == 3.0
    summary = DemoValidator.summarize([comparison])
    assert summary["count"] == 1
    assert round(summary["mean_abs_slippage_points"], 6) == 3.0
