from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from matamaple_trader.labels import LabelContract, TaskType
from matamaple_trader.validation.holdout import training_gate

@dataclass(frozen=True, slots=True)
class TrainingGuard:
    holdout_registry_path: str | Path
    shared_pipeline_exists: bool = True
    leakage_framework_exists: bool = True

    def assert_allowed(self) -> None:
        path = Path(self.holdout_registry_path)
        frozen = False
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            required = {"holdout_start", "holdout_end", "freeze_timestamp", "policy_version"}
            frozen = required.issubset(payload)
        training_gate(
            shared_pipeline_exists=self.shared_pipeline_exists,
            leakage_framework_exists=self.leakage_framework_exists,
            holdout_frozen=frozen,
        )

@dataclass(frozen=True, slots=True)
class ModelOutput:
    raw_score: np.ndarray
    probability: np.ndarray | None
    classes: np.ndarray | None
    model_name: str
    label_version: str

class ModelContractError(ValueError):
    pass

def validate_contract_for_classifier(contract: LabelContract) -> None:
    if contract.target_type not in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
        raise ModelContractError(f"classifier cannot train target_type={contract.target_type}")
