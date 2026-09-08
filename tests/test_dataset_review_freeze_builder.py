from datetime import UTC, datetime, timedelta
import json

import pandas as pd
import pytest

from matamaple_trader.data.development_builder import build_development_dataset
from matamaple_trader.labels import n_bar_direction_contract
from matamaple_trader.validation.dataset_review import review_collection_manifest
from matamaple_trader.validation.freeze_holdout_cli import main as freeze_main
from matamaple_trader.validation.holdout import HoldoutPolicy, HoldoutRegistry


def _manifest(tmp_path, *, quality_ok=True):
    path=tmp_path/'collection_manifest.json'
    payload={
        'datasets':[
            {
                'symbol':'EURUSD','timeframe':'M15','quality_ok':quality_ok,'bar_count':1000,
                'first_available_timestamp':'2024-01-01T00:00:00+00:00',
                'last_available_timestamp':'2026-09-01T00:00:00+00:00',
                'dataset_sha256':'abc' if quality_ok else None,
                'parquet_path':'data/historical/EURUSD/M15.parquet' if quality_ok else None,
                'quality_failures':[] if quality_ok else ['duplicate_timestamps'],
            },
            {
                'symbol':'GBPUSD','timeframe':'M15','quality_ok':True,'bar_count':1000,
                'first_available_timestamp':'2024-02-01T00:00:00+00:00',
                'last_available_timestamp':'2026-08-15T00:00:00+00:00',
                'dataset_sha256':'def','parquet_path':'data/historical/GBPUSD/M15.parquet','quality_failures':[],
            },
        ]
    }
    path.write_text(json.dumps(payload),encoding='utf-8')
    return path


def test_dataset_review_requires_all_clean_artifacts(tmp_path):
    good=review_collection_manifest(_manifest(tmp_path))
    assert good.ready_for_holdout_proposal
    assert good.dataset_count==2
    assert good.common_start==datetime(2024,2,1,tzinfo=UTC)
    assert good.common_end==datetime(2026,8,15,tzinfo=UTC)

    bad_path=tmp_path/'bad_manifest.json'
    bad_path.write_text(_manifest(tmp_path,quality_ok=False).read_text(),encoding='utf-8')
    bad=review_collection_manifest(bad_path)
    assert not bad.ready_for_holdout_proposal
    assert any('quality_failed:EURUSD:M15' in x for x in bad.reasons)


def test_freeze_cli_requires_explicit_confirmation_and_never_overwrites(tmp_path):
    manifest=_manifest(tmp_path)
    registry=tmp_path/'research_holdout.json'
    with pytest.raises(SystemExit):
        freeze_main([
            '--manifest',str(manifest),'--registry',str(registry),
            '--holdout-days','180','--minimum-development-days','365','--policy-version','research-v1',
        ])
    assert not registry.exists()

    rc=freeze_main([
        '--manifest',str(manifest),'--registry',str(registry),
        '--holdout-days','180','--minimum-development-days','365','--policy-version','research-v1','--confirm-freeze',
    ])
    assert rc==0 and registry.exists()
    with pytest.raises(RuntimeError,match='already frozen'):
        freeze_main([
            '--manifest',str(manifest),'--registry',str(registry),
            '--holdout-days','180','--minimum-development-days','365','--policy-version','research-v2','--confirm-freeze',
        ])


def test_development_builder_cuts_holdout_before_features_and_labels(tmp_path, monkeypatch):
    registry=tmp_path/'research_holdout.json'
    cutoff=datetime(2026,7,1,tzinfo=UTC)
    HoldoutRegistry(registry).freeze(HoldoutPolicy(
        cutoff,datetime(2026,8,1,tzinfo=UTC),datetime(2026,6,1,tzinfo=UTC),'research-v1'
    ))
    start=datetime(2026,6,1,tzinfo=UTC)
    ts=[start+timedelta(hours=i) for i in range(24*45)]
    close=[100+i*0.01 for i in range(len(ts))]
    frame=pd.DataFrame({
        'timestamp':ts,'open':close,'high':[x+0.1 for x in close],
        'low':[x-0.1 for x in close],'close':close,
    })
    monkeypatch.setattr(pd,'read_parquet',lambda _:frame.copy())
    built=build_development_dataset(
        'unused.parquet',holdout_registry_path=registry,
        label_contract=n_bar_direction_contract(horizon_bars=4,neutral_threshold=0.001),
    )
    assert (built.frame['timestamp'] < pd.Timestamp(cutoff)).all()
    assert built.frame['timestamp'].max() < pd.Timestamp(cutoff)
    assert built.label_column in built.frame.columns
    assert all(not c.startswith('label') for c in built.feature_columns)
