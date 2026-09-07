from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from matamaple_trader.labels import LabelContract
from .common import ModelOutput, TrainingGuard, validate_contract_for_classifier

class LogisticBaseline:
    def __init__(self, contract: LabelContract, *, random_state: int = 42) -> None:
        validate_contract_for_classifier(contract)
        self.contract = contract
        self.estimator = LogisticRegression(max_iter=1000, random_state=random_state)

    def fit(self, X, y, *, guard: TrainingGuard) -> "LogisticBaseline":
        guard.assert_allowed()
        self.estimator.fit(X, y)
        return self

    def predict_output(self, X) -> ModelOutput:
        probability = self.estimator.predict_proba(X)
        raw = np.asarray(self.estimator.decision_function(X))
        return ModelOutput(raw, probability, self.estimator.classes_.copy(), "logistic_regression", self.contract.label_version)
