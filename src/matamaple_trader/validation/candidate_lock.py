from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CandidateLock:
    candidate_id: str
    model_name: str
    label_name: str
    label_version: str
    horizon_bars: int
    neutral_threshold: float
    feature_version: str
    calibration_method: str
    purged_splits: int
    embargo_bars: int
    booster_n_estimators: int
    development_report_sha256: str
    pipeline_version: str
    locked_at: datetime

    def __post_init__(self) -> None:
        if self.model_name not in {'logistic_regression','xgboost','lightgbm'}:
            raise ValueError('unsupported candidate model')
        if self.horizon_bars <= 0 or self.purged_splits < 2 or self.booster_n_estimators <= 0:
            raise ValueError('invalid candidate validation configuration')
        if self.locked_at.tzinfo is None:
            raise ValueError('locked_at must be timezone-aware')
        if len(self.development_report_sha256) != 64:
            raise ValueError('development_report_sha256 must be SHA-256 hex')


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class CandidateLockRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path=Path(path)

    @staticmethod
    def _digest(payload: dict) -> str:
        clean={k:v for k,v in payload.items() if k!='integrity_sha256'}
        raw=json.dumps(clean,sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(raw).hexdigest()

    def freeze(self, lock: CandidateLock) -> None:
        if self.path.exists():
            raise RuntimeError('candidate already locked; never overwrite before holdout evaluation')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        payload=asdict(lock)
        payload['locked_at']=lock.locked_at.astimezone(UTC).isoformat()
        payload['integrity_sha256']=self._digest(payload)
        self.path.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')

    def load(self) -> dict:
        payload=json.loads(self.path.read_text(encoding='utf-8'))
        if payload.get('integrity_sha256') != self._digest(payload):
            raise RuntimeError('candidate lock integrity check failed')
        return payload
