from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from matamaple_trader.data.development_builder import build_development_dataset
from matamaple_trader.features import FeatureEngine
from matamaple_trader.labels import n_bar_direction_contract
from matamaple_trader.models import LightGBMModel, LogisticBaseline, TrainingGuard, XGBoostModel
from matamaple_trader.models.calibration import BinaryProbabilityCalibrator, CalibrationMethod
from matamaple_trader.validation.candidate_lock import CandidateLockRegistry
from matamaple_trader.validation.holdout import HoldoutRegistry
from matamaple_trader.validation.purged import PurgedKFold, PurgedSplitConfig


@dataclass(frozen=True, slots=True)
class HoldoutMetrics:
    count:int
    precision:float
    recall:float
    f1:float
    brier:float


@dataclass(frozen=True, slots=True)
class HoldoutEvaluationReport:
    candidate_id:str
    pipeline_version:str
    holdout_start:str
    holdout_end:str
    evaluated_at:str
    metrics:HoldoutMetrics
    development_rows:int
    holdout_rows:int
    calibration_source:str='development_oof_only'
    tuning_allowed_after_evaluation:bool=False
    source_partition:str='frozen_holdout'


def _factory(lock:dict, contract):
    name=lock['model_name']
    if name=='logistic_regression': return LogisticBaseline(contract)
    if name=='xgboost': return XGBoostModel(contract,n_estimators=int(lock['booster_n_estimators']),max_depth=4,learning_rate=0.05)
    if name=='lightgbm': return LightGBMModel(contract,n_estimators=int(lock['booster_n_estimators']),max_depth=-1,learning_rate=0.05)
    raise RuntimeError('unsupported locked model')


def _positive_probability(output)->np.ndarray:
    if output.probability is None or output.classes is None:
        raise RuntimeError('binary candidate must expose probability/classes')
    classes=np.asarray(output.classes)
    if set(classes.tolist())!={0,1}:
        raise RuntimeError(f'locked binary candidate classes must be {{0,1}}, got {classes.tolist()}')
    idx=int(np.where(classes==1)[0][0])
    return np.asarray(output.probability)[:,idx]


def _metrics(y,p)->HoldoutMetrics:
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=(p>=0.5).astype(int)
    return HoldoutMetrics(len(y),float(precision_score(y,pred,zero_division=0)),float(recall_score(y,pred,zero_division=0)),float(f1_score(y,pred,zero_division=0)),float(brier_score_loss(y,p)))


def _hash_payload(payload:dict)->str:
    raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()


def evaluate_frozen_holdout_once(
    parquet_path:str|Path,
    *,
    holdout_registry_path:str|Path,
    candidate_lock_path:str|Path,
    output_path:str|Path,
)->HoldoutEvaluationReport:
    out=Path(output_path)
    if out.exists():
        raise RuntimeError('frozen holdout already evaluated for this output; report is immutable')
    policy=HoldoutRegistry(holdout_registry_path).load()
    lock=CandidateLockRegistry(candidate_lock_path).load()
    if lock['label_name']!='n_bar_direction' or float(lock['neutral_threshold'])!=0.0:
        raise RuntimeError('holdout evaluator v1 supports locked binary n_bar_direction only')
    contract=n_bar_direction_contract(horizon_bars=int(lock['horizon_bars']),neutral_threshold=0.0)
    if contract.label_version!=lock['label_version']:
        raise RuntimeError('candidate lock label version mismatch')

    development=build_development_dataset(parquet_path,holdout_registry_path=holdout_registry_path,label_contract=contract,feature_engine=FeatureEngine())
    guard=TrainingGuard(holdout_registry_path); guard.assert_allowed()
    X=development.frame[list(development.feature_columns)].replace([np.inf,-np.inf],np.nan)
    y=development.frame[development.label_column]
    mask=X.notna().all(axis=1)&y.notna(); X=X.loc[mask].reset_index(drop=True); y=y.loc[mask].astype(int).reset_index(drop=True)
    if set(y.unique().tolist())!={0,1}:
        raise RuntimeError('development partition must contain both classes before holdout evaluation')

    n=len(X); event_end=np.minimum(np.arange(n)+contract.horizon_bars,n-1)
    oof=np.full(n,np.nan,dtype=float)
    splitter=PurgedKFold(PurgedSplitConfig(n_splits=int(lock['purged_splits']),embargo_bars=int(lock['embargo_bars'])))
    for train_idx,test_idx in splitter.split(X,event_end_index=event_end):
        if set(y.iloc[train_idx].unique().tolist())!={0,1}:
            raise RuntimeError('single-class development fold blocks locked calibration')
        model=_factory(lock,contract).fit(X.iloc[train_idx],y.iloc[train_idx],guard=guard)
        oof[test_idx]=_positive_probability(model.predict_output(X.iloc[test_idx]))
    if not np.isfinite(oof).all(): raise RuntimeError('development OOF calibration coverage incomplete')
    method=CalibrationMethod(str(lock['calibration_method']))
    calibrator=BinaryProbabilityCalibrator(method)
    calibrator.fit(oof,y.to_numpy(),source='oof')

    bars=pd.read_parquet(parquet_path).copy()
    bars['timestamp']=pd.to_datetime(bars['timestamp'],utc=True)
    start=pd.Timestamp(datetime.fromisoformat(policy['holdout_start']).astimezone(UTC))
    end=pd.Timestamp(datetime.fromisoformat(policy['holdout_end']).astimezone(UTC))
    bars=bars.loc[bars['timestamp']<end].copy()
    featured=FeatureEngine().transform(bars)
    featured[development.label_column]=featured['close'].shift(-contract.horizon_bars)/featured['close']-1.0
    featured[development.label_column]=(featured[development.label_column]>0).astype(float).where(featured['close'].shift(-contract.horizon_bars).notna())
    hold=featured.loc[(featured['timestamp']>=start)&(featured['timestamp']<end)].copy()
    HX=hold[list(development.feature_columns)].replace([np.inf,-np.inf],np.nan); Hy=hold[development.label_column]
    hmask=HX.notna().all(axis=1)&Hy.notna(); HX=HX.loc[hmask]; Hy=Hy.loc[hmask].astype(int)
    if len(HX)==0: raise RuntimeError('no evaluable frozen holdout rows')

    final_model=_factory(lock,contract).fit(X,y,guard=guard)
    rawp=_positive_probability(final_model.predict_output(HX))
    calibrated=calibrator.transform(rawp)
    report=HoldoutEvaluationReport(lock['candidate_id'],lock['pipeline_version'],start.isoformat(),end.isoformat(),datetime.now(UTC).isoformat(),_metrics(Hy,calibrated),len(X),len(HX))
    payload=asdict(report); payload['integrity_sha256']=_hash_payload(payload)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    return report
