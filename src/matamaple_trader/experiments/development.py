from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from matamaple_trader.data.development_builder import DevelopmentDataset
from matamaple_trader.labels import TaskType
from matamaple_trader.models import LightGBMModel, LogisticBaseline, TrainingGuard, XGBoostModel
from matamaple_trader.models.calibration import BinaryProbabilityCalibrator, CalibrationMethod
from matamaple_trader.validation.purged import CombinatorialPurgedCV, PurgedKFold, PurgedSplitConfig
from matamaple_trader.validation.walk_forward import ExpandingWalkForward, WalkForwardConfig


@dataclass(frozen=True, slots=True)
class MetricSet:
    count: int
    precision: float
    recall: float
    f1: float
    brier: float


@dataclass(frozen=True, slots=True)
class ModelDevelopmentResult:
    model_name: str
    purged_oof: MetricSet
    calibrated_oof_brier: float
    calibration_method: str
    walk_forward: MetricSet
    purged_folds: int
    cpcv_paths: int
    walk_forward_windows: int


@dataclass(frozen=True, slots=True)
class DevelopmentExperimentReport:
    label_version: str
    target_type: str
    feature_columns: tuple[str, ...]
    development_rows: int
    cutoff_timestamp: str
    results: tuple[ModelDevelopmentResult, ...]
    selected_model: str
    selection_rule: str
    source_partition: str = "development_only"


def _clean_matrix(dataset: DevelopmentDataset):
    frame = dataset.frame.copy()
    cols = list(dataset.feature_columns)
    X = frame[cols].replace([np.inf, -np.inf], np.nan)
    y = frame[dataset.label_column]
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)
    if len(X) < 30:
        raise RuntimeError("not enough clean development rows for experiment")
    return X, y


def _positive_probability(output) -> np.ndarray:
    if output.probability is None or output.classes is None:
        raise RuntimeError("binary classifier must expose probabilities and classes")
    classes = np.asarray(output.classes)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise RuntimeError(f"binary experiment requires classes {{0,1}}, got {classes.tolist()}")
    positive = int(np.where(classes == 1)[0][0])
    return np.asarray(output.probability)[:, positive]


def _metrics(y_true, probability) -> MetricSet:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= 0.5).astype(int)
    return MetricSet(
        count=len(y),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        brier=float(brier_score_loss(y, p)),
    )


def _factories(contract):
    # Baseline is intentionally first. Advanced models must beat it OOS to justify complexity.
    return (
        ("logistic_regression", lambda: LogisticBaseline(contract)),
        ("xgboost", lambda: XGBoostModel(contract, n_estimators=100, max_depth=4, learning_rate=0.05)),
        ("lightgbm", lambda: LightGBMModel(contract, n_estimators=100, max_depth=-1, learning_rate=0.05)),
    )


def run_development_experiment(
    dataset: DevelopmentDataset,
    *,
    label_contract,
    holdout_registry_path: str | Path,
    purged_splits: int = 5,
    embargo_bars: int = 0,
    cpcv_groups: int = 6,
    cpcv_test_groups: int = 2,
    wf_min_train_bars: int = 200,
    wf_test_bars: int = 50,
    wf_step_bars: int | None = None,
    calibration_method: CalibrationMethod = CalibrationMethod.PLATT,
) -> DevelopmentExperimentReport:
    """Run model research on the development partition only.

    This function never loads historical Parquet or holdout outcomes. It accepts an already
    isolated DevelopmentDataset and uses the frozen registry only as a training authorization
    guard. Splits are chronological/purged; shuffled train/test splitting is not supported.
    """
    if label_contract.target_type != TaskType.BINARY_CLASSIFICATION:
        raise ValueError("development experiment v1 calibration supports binary classification only")
    if dataset.cutoff_timestamp.isoformat() == "":
        raise ValueError("dataset cutoff is required")

    X, y = _clean_matrix(dataset)
    n = len(X)
    if wf_min_train_bars + wf_test_bars > n:
        raise ValueError("walk-forward configuration exceeds development rows")

    guard = TrainingGuard(holdout_registry_path)
    guard.assert_allowed()
    event_end = np.minimum(np.arange(n) + label_contract.horizon_bars, n - 1)

    purged = PurgedKFold(PurgedSplitConfig(n_splits=purged_splits, embargo_bars=embargo_bars))
    cpcv = CombinatorialPurgedCV(n_groups=cpcv_groups, test_groups=cpcv_test_groups, embargo_bars=embargo_bars)
    cpcv_paths = sum(1 for _ in cpcv.split(X, event_end_index=event_end))

    wf = ExpandingWalkForward(WalkForwardConfig(
        min_train_bars=wf_min_train_bars,
        test_bars=wf_test_bars,
        step_bars=wf_step_bars,
        gap_bars=label_contract.horizon_bars,
    ))

    results: list[ModelDevelopmentResult] = []
    for model_name, factory in _factories(label_contract):
        oof_p = np.full(n, np.nan, dtype=float)
        purged_folds = 0
        for train_idx, test_idx in purged.split(X, event_end_index=event_end):
            if len(train_idx) == 0:
                raise RuntimeError("purged split produced empty training set")
            model = factory().fit(X.iloc[train_idx], y.iloc[train_idx], guard=guard)
            oof_p[test_idx] = _positive_probability(model.predict_output(X.iloc[test_idx]))
            purged_folds += 1
        valid = np.isfinite(oof_p)
        if not valid.all():
            raise RuntimeError("purged OOF predictions did not cover every development row")
        purged_metrics = _metrics(y, oof_p)

        calibrator = BinaryProbabilityCalibrator(calibration_method)
        calibration = calibrator.fit(oof_p, y.to_numpy(), source="oof")

        wf_y: list[int] = []
        wf_p: list[float] = []
        wf_windows = 0
        for train_idx, test_idx in wf.split(X):
            model = factory().fit(X.iloc[train_idx], y.iloc[train_idx], guard=guard)
            probability = _positive_probability(model.predict_output(X.iloc[test_idx]))
            wf_y.extend(y.iloc[test_idx].tolist())
            wf_p.extend(probability.tolist())
            wf_windows += 1
        if not wf_y:
            raise RuntimeError("walk-forward produced no evaluation windows")
        wf_metrics = _metrics(wf_y, wf_p)

        results.append(ModelDevelopmentResult(
            model_name=model_name,
            purged_oof=purged_metrics,
            calibrated_oof_brier=calibration.brier_after,
            calibration_method=calibration.method.value,
            walk_forward=wf_metrics,
            purged_folds=purged_folds,
            cpcv_paths=cpcv_paths,
            walk_forward_windows=wf_windows,
        ))

    # Selection is development-only: prioritize WF F1, then lower WF Brier, then lower calibrated OOF Brier.
    selected = max(results, key=lambda r: (r.walk_forward.f1, -r.walk_forward.brier, -r.calibrated_oof_brier))
    return DevelopmentExperimentReport(
        label_version=label_contract.label_version,
        target_type=label_contract.target_type.value,
        feature_columns=dataset.feature_columns,
        development_rows=n,
        cutoff_timestamp=dataset.cutoff_timestamp.isoformat(),
        results=tuple(results),
        selected_model=selected.model_name,
        selection_rule="max walk_forward F1, then min walk_forward Brier, then min calibrated OOF Brier; holdout excluded",
    )


def write_experiment_report(report: DevelopmentExperimentReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return out
