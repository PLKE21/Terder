from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from matamaple_trader.labels import LabelContract, TaskType
from .common import ModelOutput, TrainingGuard

class LightGBMModel:
    def __init__(self, contract: LabelContract, *, random_state: int = 42, n_estimators: int = 100, max_depth: int = -1, learning_rate: float = 0.05) -> None:
        self.contract = contract
        self._class_values: np.ndarray | None = None
        common = dict(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=1, verbosity=-1)
        if contract.target_type == TaskType.REGRESSION:
            self.estimator = LGBMRegressor(**common)
        elif contract.target_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            self.estimator = LGBMClassifier(**common)
        else:
            raise ValueError(f"unsupported LightGBM task: {contract.target_type}")

    def fit(self, X, y, *, guard: TrainingGuard) -> "LightGBMModel":
        guard.assert_allowed()
        y_arr=np.asarray(y)
        if self.contract.target_type == TaskType.MULTICLASS_CLASSIFICATION:
            classes=np.unique(y_arr)
            expected=np.asarray([-1,0,1])
            if not np.array_equal(classes, expected):
                raise ValueError("multiclass contract requires observed classes exactly [-1, 0, 1]")
            self._class_values=expected
            mapping={-1:0,0:1,1:2}
            encoded=np.asarray([mapping[int(v)] for v in y_arr],dtype=int)
            self.estimator.fit(X,encoded)
        else:
            self.estimator.fit(X,y_arr)
            if self.contract.target_type == TaskType.BINARY_CLASSIFICATION:
                self._class_values=np.asarray(self.estimator.classes_)
        return self

    def predict_output(self, X) -> ModelOutput:
        if self.contract.target_type == TaskType.REGRESSION:
            raw=np.asarray(self.estimator.predict(X))
            return ModelOutput(raw,None,None,"lightgbm",self.contract.label_version)

        raw=np.asarray(self.estimator.predict(X,raw_score=True))
        proba=np.asarray(self.estimator.predict_proba(X))
        classes=self._class_values.copy() if self._class_values is not None else np.asarray(self.estimator.classes_)
        return ModelOutput(raw,proba,classes,"lightgbm",self.contract.label_version)
