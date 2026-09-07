from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from matamaple_trader.labels import LabelContract, TaskType
from .common import ModelOutput, TrainingGuard

class LightGBMModel:
    def __init__(self, contract: LabelContract, *, random_state: int = 42, n_estimators: int = 100, max_depth: int = -1, learning_rate: float = 0.05) -> None:
        self.contract = contract
        common = dict(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=1, verbosity=-1)
        if contract.target_type == TaskType.REGRESSION:
            self.estimator = LGBMRegressor(**common)
        elif contract.target_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            self.estimator = LGBMClassifier(**common)
        else:
            raise ValueError(f"unsupported LightGBM task: {contract.target_type}")

    def fit(self, X, y, *, guard: TrainingGuard) -> "LightGBMModel":
        guard.assert_allowed()
        self.estimator.fit(X, y)
        return self

    def predict_output(self, X) -> ModelOutput:
        pred = np.asarray(self.estimator.predict(X))
        if hasattr(self.estimator, "predict_proba"):
            proba = np.asarray(self.estimator.predict_proba(X))
            classes = np.asarray(self.estimator.classes_)
        else:
            proba = None
            classes = None
        return ModelOutput(pred, proba, classes, "lightgbm", self.contract.label_version)
