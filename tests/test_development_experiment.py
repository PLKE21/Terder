from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from matamaple_trader.data.development_builder import DevelopmentDataset
from matamaple_trader.experiments import run_development_experiment
from matamaple_trader.labels import n_bar_direction_contract
from matamaple_trader.validation.holdout import HoldoutPolicy, HoldoutRegistry


def frozen_registry(tmp_path, cutoff):
    path=tmp_path/'research_holdout.json'
    HoldoutRegistry(path).freeze(HoldoutPolicy(
        holdout_start=cutoff,
        holdout_end=cutoff+timedelta(days=180),
        freeze_timestamp=cutoff-timedelta(days=1),
        policy_version='research-holdout-v1',
    ))
    return path


def development_dataset(cutoff, *, leak=False):
    n=96
    start=cutoff-timedelta(minutes=15*n)
    ts=pd.date_range(start=start,periods=n,freq='15min',tz='UTC')
    if leak:
        ts=ts[:-1].append(pd.DatetimeIndex([pd.Timestamp(cutoff)]))
    x=np.linspace(-3,3,n)
    labels=(np.arange(n)%2).astype(int)
    frame=pd.DataFrame({
        'timestamp':ts,
        'feature_a':x,
        'feature_b':np.sin(np.arange(n)/5),
        'label__n_bar_direction__nbar-direction-v1':labels,
    })
    return DevelopmentDataset(
        frame=frame,
        feature_columns=('feature_a','feature_b'),
        label_column='label__n_bar_direction__nbar-direction-v1',
        cutoff_timestamp=cutoff,
    )


def test_development_experiment_runs_purged_cpcv_walk_forward_without_holdout(tmp_path):
    cutoff=datetime(2026,7,1,tzinfo=UTC)
    registry=frozen_registry(tmp_path,cutoff)
    dataset=development_dataset(cutoff)
    contract=n_bar_direction_contract(horizon_bars=2,neutral_threshold=0.0)

    report=run_development_experiment(
        dataset,
        label_contract=contract,
        holdout_registry_path=registry,
        purged_splits=3,
        cpcv_groups=4,
        cpcv_test_groups=1,
        wf_min_train_bars=40,
        wf_test_bars=10,
        wf_step_bars=10,
        booster_n_estimators=5,
    )

    assert report.source_partition=='development_only'
    assert report.cutoff_timestamp==cutoff.isoformat()
    assert {r.model_name for r in report.results}=={'logistic_regression','xgboost','lightgbm'}
    assert all(r.purged_folds==3 for r in report.results)
    assert all(r.cpcv_paths==4 for r in report.results)
    assert all(r.walk_forward_windows>0 for r in report.results)
    assert report.selected_model in {'logistic_regression','xgboost','lightgbm'}
    assert 'holdout excluded' in report.selection_rule


def test_development_experiment_rejects_holdout_timestamp_inside_dataset(tmp_path):
    cutoff=datetime(2026,7,1,tzinfo=UTC)
    registry=frozen_registry(tmp_path,cutoff)
    dataset=development_dataset(cutoff,leak=True)
    contract=n_bar_direction_contract(horizon_bars=2,neutral_threshold=0.0)

    with pytest.raises(RuntimeError,match='holdout leakage guard'):
        run_development_experiment(
            dataset,
            label_contract=contract,
            holdout_registry_path=registry,
            purged_splits=3,
            cpcv_groups=4,
            cpcv_test_groups=1,
            wf_min_train_bars=40,
            wf_test_bars=10,
            booster_n_estimators=2,
        )


def test_development_experiment_rejects_cutoff_mismatch(tmp_path):
    cutoff=datetime(2026,7,1,tzinfo=UTC)
    registry=frozen_registry(tmp_path,cutoff)
    dataset=development_dataset(cutoff+timedelta(days=1))
    contract=n_bar_direction_contract(horizon_bars=2,neutral_threshold=0.0)

    with pytest.raises(RuntimeError,match='cutoff does not match'):
        run_development_experiment(
            dataset,
            label_contract=contract,
            holdout_registry_path=registry,
            purged_splits=3,
            cpcv_groups=4,
            cpcv_test_groups=1,
            wf_min_train_bars=40,
            wf_test_bars=10,
            booster_n_estimators=2,
        )
