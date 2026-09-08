from datetime import UTC, datetime, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from matamaple_trader.domain import OperationalState, Signal, SignalResult
from matamaple_trader.shadow.store import ShadowStore
from matamaple_trader.validation.candidate_lock import CandidateLock, CandidateLockRegistry
from matamaple_trader.validation.holdout import HoldoutPolicy, HoldoutRegistry
from matamaple_trader.validation.holdout_evaluation import evaluate_frozen_holdout_once
from matamaple_trader.validation.release_readiness import release_readiness


def _result(ts,symbol='EURUSD'):
    return SignalResult(timestamp=ts,symbol=symbol,signal=Signal.WAIT,raw_score=0.1,calibrated_probability=0.55,regime='RANGE',entry_low=None,entry_high=None,stop_loss=None,tp1=None,tp2=None,tp3=None,risk_reward=None,grade='C',spread_state='NORMAL',portfolio_warning=None,pipeline_version='v0.1.0',operational_state=OperationalState.ACTIVE)


def test_candidate_lock_is_hash_protected(tmp_path):
    p=tmp_path/'candidate.json'
    lock=CandidateLock('logistic_regression:nbar-direction-v1:v0.1.0','logistic_regression','n_bar_direction','nbar-direction-v1',2,0.0,'features-v1','PLATT',3,0,2,'a'*64,'v0.1.0',datetime.now(UTC))
    CandidateLockRegistry(p).freeze(lock)
    payload=json.loads(p.read_text()); payload['model_name']='xgboost'; p.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError,match='integrity'):
        CandidateLockRegistry(p).load()


def test_shadow_foreign_key_and_release_evidence_gates(tmp_path):
    db=tmp_path/'shadow.sqlite'
    store=ShadowStore(db)
    with pytest.raises(RuntimeError):
        store.attach_outcome(999,{'r':1})
    now=datetime.now(UTC)
    for i in range(3):
        pid=store.append_prediction(_result(now+timedelta(minutes=i)))
        store.attach_outcome(pid,{'r':0.1})
    holdout=tmp_path/'holdout_report.json'; holdout.write_text('{}')
    demo=tmp_path/'demo.json'; demo.write_text(json.dumps({'count':3,'mean_abs_slippage_points':1.2}))
    report=release_readiness(holdout_report_path=holdout,shadow_db_path=db,demo_summary_path=demo,min_shadow_predictions=3,min_shadow_outcome_coverage=1.0,min_manual_fills=3,max_mean_abs_slippage_points=2.0)
    assert report['validated_for_real_use'] is True
    assert report['auto_trading_enabled'] is False


def test_release_gate_fails_safe_without_real_evidence(tmp_path):
    report=release_readiness(holdout_report_path=tmp_path/'missing.json',shadow_db_path=tmp_path/'shadow.sqlite',demo_summary_path=tmp_path/'demo.json',min_shadow_predictions=1,min_manual_fills=1)
    assert report['code_path_complete'] is True
    assert report['validated_for_real_use'] is False
    assert 'missing_frozen_holdout_evaluation' in report['reasons']


def test_frozen_holdout_is_one_shot_and_development_calibrated(tmp_path,monkeypatch):
    n=260
    ts=pd.date_range('2026-01-01',periods=n,freq='15min',tz='UTC')
    close=100+0.02*np.arange(n)+0.6*np.sin(np.arange(n)*1.7)
    frame=pd.DataFrame({'timestamp':ts,'open':close-0.02,'high':close+0.08,'low':close-0.08,'close':close,'tick_volume':100,'spread':10})
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:frame.copy())

    cutoff=ts[190].to_pydatetime(); end=(ts[-1]+pd.Timedelta(minutes=15)).to_pydatetime()
    holdreg=tmp_path/'research_holdout.json'
    HoldoutRegistry(holdreg).freeze(HoldoutPolicy(cutoff,end,cutoff-timedelta(days=1),'research-holdout-v1'))
    lockpath=tmp_path/'candidate.json'
    CandidateLockRegistry(lockpath).freeze(CandidateLock('logistic_regression:nbar-direction-v1:v0.1.0','logistic_regression','n_bar_direction','nbar-direction-v1',2,0.0,'features-v1','PLATT',3,0,2,'b'*64,'v0.1.0',datetime.now(UTC)))
    out=tmp_path/'holdout_eval.json'
    report=evaluate_frozen_holdout_once('ignored.parquet',holdout_registry_path=holdreg,candidate_lock_path=lockpath,output_path=out)
    assert report.source_partition=='frozen_holdout'
    assert report.calibration_source=='development_oof_only'
    assert report.tuning_allowed_after_evaluation is False
    assert report.metrics.count>0
    payload=json.loads(out.read_text())
    assert len(payload['integrity_sha256'])==64
    with pytest.raises(RuntimeError,match='already evaluated'):
        evaluate_frozen_holdout_once('ignored.parquet',holdout_registry_path=holdreg,candidate_lock_path=lockpath,output_path=out)
