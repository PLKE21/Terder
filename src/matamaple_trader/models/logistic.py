from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matamaple_trader.labels import LabelContract
from .common import ModelOutput, TrainingGuard, validate_contract_for_classifier

class LogisticBaseline:
    def __init__(self, contract: LabelContract, *, random_state: int = 42) -> None:
        validate_contract_for_classifier(contract)
        self.contract = contract
        self.estimator = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=random_state)),
        ])

    def fit(self, X, y, *, guard: TrainingGuard) -> "LogisticBaseline":
        guard.assert_allowed()
        self.estimator.fit(X, y)
        return self

    def predict_output(self, X) -> ModelOutput:
        probability = np.asarray(self.estimator.predict_proba(X))
        raw = np.asarray(self.estimator.decision_function(X))
        classes = np.asarray(self.estimator.named_steps['model'].classes_).copy()
        return ModelOutput(raw, probability, classes, "logistic_regression", self.contract.label_version)
