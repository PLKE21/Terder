from datetime import UTC, datetime
import numpy as np
import pandas as pd
import pytest

from matamaple_trader.features import FeatureConfig, FeatureEngine, align_closed_higher_timeframe
from matamaple_trader.labels import TaskType, forward_return_contract, make_labels, n_bar_direction_contract, triple_barrier_contract
from matamaple_trader.models import LightGBMModel, LogisticBaseline, TrainingGuard, XGBoostModel
from matamaple_trader.validation.holdout import HoldoutPolicy, HoldoutRegistry


def bars(n=80):
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.arange(n) * 0.1 + np.sin(np.arange(n) / 3)
    return pd.DataFrame({"timestamp": ts, "open": close - 0.1, "high": close + 0.4, "low": close - 0.5, "close": close})


def frozen_guard(tmp_path):
    path = tmp_path / "research_holdout.json"
    HoldoutRegistry(path).freeze(HoldoutPolicy(
        holdout_start=datetime(2025, 7, 1, tzinfo=UTC),
        holdout_end=datetime(2026, 1, 1, tzinfo=UTC),
        freeze_timestamp=datetime(2025, 6, 1, tzinfo=UTC),
        policy_version="holdout-v1",
    ))
    return TrainingGuard(path)


def test_feature_engine_is_causal_for_past_rows():
    engine = FeatureEngine(FeatureConfig(atr_window=3, rsi_window=3, volatility_window=3))
    first = engine.transform(bars(30))
    extended = engine.transform(bars(40))
    cols = ["return_1", "atr", "rsi", "ema_spread_pct", "volatility"]
    pd.testing.assert_frame_equal(first[cols], extended.iloc[:30][cols])


def test_higher_timeframe_is_unavailable_until_close():
    base = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01T10:15Z", "2026-01-01T11:00Z"]), "x": [1, 2]})
    higher = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01T10:00Z"]), "close": [123.0]})
    out = align_closed_higher_timeframe(base, higher, higher_bar_seconds=3600, suffix="h1")
    assert np.isnan(out.loc[0, "close_h1"])
    assert out.loc[1, "close_h1"] == 123.0


def test_label_contracts_are_explicit_and_not_interchangeable():
    assert forward_return_contract().target_type == TaskType.REGRESSION
    assert n_bar_direction_contract(neutral_threshold=0).target_type == TaskType.BINARY_CLASSIFICATION
    assert n_bar_direction_contract(neutral_threshold=0.001).target_type == TaskType.MULTICLASS_CLASSIFICATION
    assert triple_barrier_contract().supports_three_class_probability


def test_label_generation_leaves_future_tail_unknown():
    close = pd.Series([100, 101, 102, 103, 104, 105], dtype=float)
    y = make_labels(close, forward_return_contract(horizon_bars=2))
    assert y.iloc[-2:].isna().all()


def test_training_is_blocked_without_frozen_holdout(tmp_path):
    model = LogisticBaseline(n_bar_direction_contract())
    with pytest.raises(RuntimeError, match="frozen_holdout"):
        model.fit([[0], [1], [2], [3]], [0, 0, 1, 1], guard=TrainingGuard(tmp_path / "missing.json"))


def test_logistic_baseline_trains_only_after_gate(tmp_path):
    X = np.array([[0.0], [0.2], [0.4], [1.0], [1.2], [1.4]])
    y = np.array([0, 0, 0, 1, 1, 1])
    out = LogisticBaseline(n_bar_direction_contract()).fit(X, y, guard=frozen_guard(tmp_path)).predict_output(X)
    assert out.probability.shape == (6, 2)
    assert out.model_name == "logistic_regression"


def test_xgboost_binary_wrapper(tmp_path):
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 20).astype(int)
    out = XGBoostModel(n_bar_direction_contract(), n_estimators=8, max_depth=2).fit(X, y, guard=frozen_guard(tmp_path)).predict_output(X[:3])
    assert out.probability.shape == (3, 2)


def test_lightgbm_regression_wrapper(tmp_path):
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = X[:, 0] * 0.01
    out = LightGBMModel(forward_return_contract(), n_estimators=8).fit(X, y, guard=frozen_guard(tmp_path)).predict_output(X[:3])
    assert out.probability is None
    assert out.raw_score.shape == (3,)
