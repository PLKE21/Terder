from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class CalibrationMethod(StrEnum):
    PLATT = "PLATT"
    ISOTONIC = "ISOTONIC"


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    brier_before: float
    brier_after: float
    method: CalibrationMethod
    calibration_version: str


class BinaryProbabilityCalibrator:
    """Fit only on OOF/independent validation predictions, never frozen holdout outcomes."""

    def __init__(self, method: CalibrationMethod = CalibrationMethod.PLATT, *, version: str = "calibration-v1") -> None:
        self.method = method
        self.version = version
        self._model = None

    def fit(self, raw_probability, y_true, *, source: str) -> CalibrationReport:
        source_norm = source.strip().lower()
        if source_norm in {"holdout", "frozen_holdout", "research_holdout", "final_holdout"}:
            raise RuntimeError("calibration must not be fit on frozen holdout outcomes")
        if source_norm not in {"oof", "validation", "independent_validation"}:
            raise ValueError("source must be OOF or independent validation predictions")
        p = np.asarray(raw_probability, dtype=float).reshape(-1)
        y = np.asarray(y_true, dtype=int).reshape(-1)
        if p.shape != y.shape:
            raise ValueError("probabilities and labels must have same shape")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must be within [0, 1]")
        before = float(brier_score_loss(y, p))
        if self.method == CalibrationMethod.PLATT:
            eps = 1e-6
            logits = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
            model = LogisticRegression(max_iter=1000)
            model.fit(logits, y)
            self._model = model
        else:
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p, y)
            self._model = model
        after = float(brier_score_loss(y, self.transform(p)))
        return CalibrationReport(before, after, self.method, self.version)

    def transform(self, raw_probability):
        if self._model is None:
            raise RuntimeError("calibrator is not fitted")
        p = np.asarray(raw_probability, dtype=float).reshape(-1)
        if self.method == CalibrationMethod.PLATT:
            eps = 1e-6
            logits = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
            return self._model.predict_proba(logits)[:, 1]
        return np.asarray(self._model.predict(p), dtype=float)
