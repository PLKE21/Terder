from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import numpy as np

from matamaple_trader.labels import LabelContract, TaskType
from matamaple_trader.validation.holdout import HoldoutRegistry, training_gate

@dataclass(frozen=True, slots=True)
class TrainingGuard:
    holdout_registry_path: str | Path
    shared_pipeline_exists: bool | None = None
    leakage_framework_exists: bool | None = None

    def assert_allowed(self) -> None:
        pipeline_ok = self.shared_pipeline_exists if self.shared_pipeline_exists is not None else find_spec('matamaple_trader.pipeline.core') is not None
        leakage_ok = self.leakage_framework_exists if self.leakage_framework_exists is not None else find_spec('matamaple_trader.validation.leakage') is not None
        frozen=False
        path=Path(self.holdout_registry_path)
        if path.is_file():
            HoldoutRegistry(path).load()  # verifies required fields and integrity hash
            frozen=True
        training_gate(
            shared_pipeline_exists=bool(pipeline_ok),
            leakage_framework_exists=bool(leakage_ok),
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
