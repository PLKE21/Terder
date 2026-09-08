import math
import numpy as np
import pytest

from matamaple_trader.execution import CostConfig, CostEngine
from matamaple_trader.execution.costs import Side
from matamaple_trader.models.calibration import BinaryProbabilityCalibrator, CalibrationMethod
from matamaple_trader.regime import MarketRegime, RegimeEngine, RegimeThresholds
from matamaple_trader.validation.purged import CombinatorialPurgedCV, PurgedKFold, PurgedSplitConfig
from matamaple_trader.validation.walk_forward import ExpandingWalkForward, WalkForwardConfig


def test_purged_kfold_removes_event_overlap_and_embargo():
    X = np.zeros((12, 1))
    ends = np.arange(12) + 2
    splits = list(PurgedKFold(PurgedSplitConfig(n_splits=3, embargo_bars=1)).split(X, event_end_index=ends))
    train, test = splits[1]
    test_start, test_end = test[0], test[-1]
    assert all(not (i <= test_end and ends[i] >= test_start) for i in train)
    assert test_end + 1 not in train


def test_cpcv_generates_group_combinations_without_overlap():
    X = np.zeros((60, 1))
    splits = list(CombinatorialPurgedCV(n_groups=6, test_groups=2, embargo_bars=1).split(X))
    assert len(splits) == math.comb(6, 2)
    for train, test in splits:
        assert not set(train).intersection(set(test))


def test_walk_forward_is_expanding_and_strictly_chronological():
    X = np.zeros((50, 1))
    splits = list(ExpandingWalkForward(WalkForwardConfig(min_train_bars=20, test_bars=5, gap_bars=2)).split(X))
    assert splits
    previous_train_size = 0
    for train, test in splits:
        assert len(train) > previous_train_size
        assert train.max() < test.min()
        assert test.min() - train.max() - 1 == 2
        previous_train_size = len(train)


def test_calibration_rejects_holdout_and_outputs_probabilities():
    raw = np.array([0.05, 0.2, 0.3, 0.7, 0.8, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])
    calibrator = BinaryProbabilityCalibrator(CalibrationMethod.PLATT)
    with pytest.raises(RuntimeError, match="frozen holdout"):
        calibrator.fit(raw, y, source="research_holdout")
    report = calibrator.fit(raw, y, source="oof")
    calibrated = calibrator.transform(raw)
    assert report.brier_before >= 0 and report.brier_after >= 0
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_regime_thresholds_cannot_be_fit_on_holdout():
    engine = RegimeEngine()
    trend = np.linspace(-2, 2, 100)
    vol = np.linspace(0.1, 1.0, 100)
    with pytest.raises(RuntimeError, match="frozen holdout"):
        engine.fit(trend, vol, source="holdout")
    engine.fit(trend, vol, source="development")
    assert engine.thresholds is not None


def test_regime_classification_is_deterministic():
    engine = RegimeEngine(RegimeThresholds(trend_abs_threshold=1.0, high_vol_threshold=0.8, low_vol_threshold=0.2))
    assert engine.classify(trend_strength=2.0, volatility=0.5) == MarketRegime.TREND
    assert engine.classify(trend_strength=0.1, volatility=0.9) == MarketRegime.HIGH_VOLATILITY
    assert engine.classify(trend_strength=0.1, volatility=0.1) == MarketRegime.LOW_VOLATILITY
    assert engine.classify(trend_strength=0.1, volatility=0.5) == MarketRegime.RANGE


def test_cost_engine_tracks_spread_slippage_commission_and_swap():
    engine = CostEngine(CostConfig(slippage_points=2, commission_per_lot=3.5, point_value_per_lot=1.0))
    entry = engine.simulate_entry(side=Side.BUY, requested_price=100.0, bid=99.99, ask=100.01, point=0.01, lots=1)
    assert entry.simulated_fill == pytest.approx(100.03)
    assert entry.costs.spread_cost == pytest.approx(2.0)
    assert entry.costs.slippage == pytest.approx(2.0)
    assert entry.costs.commission == pytest.approx(3.5)
    full = engine.full_costs(entry=entry, swap_cost=1.25)
    assert full.swap == pytest.approx(1.25)
    assert full.total == pytest.approx(8.75)


def test_gap_through_stop_uses_first_executable_quote():
    engine = CostEngine(CostConfig(point_value_per_lot=1.0))
    long_stop = engine.simulate_stop_fill(side=Side.BUY, stop_price=100.0, next_bid=99.5, next_ask=99.6, point=0.1)
    assert long_stop.simulated_fill == pytest.approx(99.5)
    assert long_stop.costs.gap_slippage == pytest.approx(5.0)
    short_stop = engine.simulate_stop_fill(side=Side.SELL, stop_price=100.0, next_bid=100.4, next_ask=100.5, point=0.1)
    assert short_stop.simulated_fill == pytest.approx(100.5)
    assert short_stop.costs.gap_slippage == pytest.approx(5.0)
