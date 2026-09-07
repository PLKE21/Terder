from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from matamaple_trader.labels import LabelContract, TaskType
from .common import ModelOutput, TrainingGuard

class XGBoostModel:
    def __init__(self, contract: LabelContract, *, random_state: int = 42, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05) -> None:
        self.contract = contract
        common = dict(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=1)
        if contract.target_type == TaskType.REGRESSION:
            self.estimator = XGBRegressor(objective="reg:squarederror", **common)
        elif contract.target_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            self.estimator = XGBClassifier(**common)
        else:
            raise ValueError(f"unsupported XGBoost task: {contract.target_type}")

    def fit(self, X, y, *, guard: TrainingGuard) -> "XGBoostModel":
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
        return ModelOutput(pred, proba, classes, "xgboost", self.contract.label_version)
