from datetime import UTC, datetime, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from matamaple_trader.broker.spec_integrity import BrokerSpecMonitor
from matamaple_trader.data.collector import HistoricalCollector
from matamaple_trader.data.integrity import validate_bars
from matamaple_trader.domain import OperationalState, SymbolSpec
from matamaple_trader.labels import n_bar_direction_contract
from matamaple_trader.models import LightGBMModel, TrainingGuard, XGBoostModel
from matamaple_trader.realtime.watcher import WatcherConfig, should_evaluate
from matamaple_trader.validation.holdout import HoldoutPolicy, HoldoutRegistry
from matamaple_trader.validation.leakage import LeakageError, assert_no_label_features


class RawAdapter:
    def get_rates(self, symbol, timeframe, start, end):
        t0=pd.Timestamp('2026-01-01T00:00:00Z')
        return pd.DataFrame({
            'timestamp':[t0+pd.Timedelta('15min'),t0,t0],
            'open':[1.0,1.0,1.0],
            'high':[2.0,2.0,2.0],
            'low':[0.5,0.5,0.5],
            'close':[1.5,1.5,1.5],
        })


def freeze(tmp_path):
    path=tmp_path/'research_holdout.json'
    HoldoutRegistry(path).freeze(HoldoutPolicy(
        datetime(2026,7,1,tzinfo=UTC),
        datetime(2026,8,1,tzinfo=UTC),
        datetime(2026,6,1,tzinfo=UTC),
        'holdout-v1',
    ))
    return path


def spec(contract=100000, sessions=()):
    return SymbolSpec('EURUSD',5,0.00001,0.00001,1.0,contract,0.01,100.0,0.01,-1.0,1.0,sessions)


def test_collector_preserves_source_duplicates_and_order_for_integrity():
    start=datetime(2026,1,1,tzinfo=UTC)
    frame,meta=HistoricalCollector(RawAdapter()).collect('EURUSD',15,'M15',start,start+timedelta(hours=1),expected_delta=pd.Timedelta('15min'))
    assert frame['timestamp'].duplicated().sum()==1
    assert not frame['timestamp'].is_monotonic_increasing
    report=validate_bars(frame,pd.Timedelta('15min'),min_bars=1)
    assert 'duplicate_timestamps' in report.failures
    assert 'time_reversal' in report.failures
    assert meta.bar_count==3


def test_integrity_detects_misaligned_timeframe_and_stale_tick():
    ts=pd.to_datetime(['2026-01-01T00:05:00Z','2026-01-01T00:20:00Z'])
    frame=pd.DataFrame({'timestamp':ts,'open':[1,1],'high':[2,2],'low':[0.5,0.5],'close':[1.5,1.5]})
    report=validate_bars(
        frame,
        pd.Timedelta('15min'),
        min_bars=1,
        latest_tick_timestamp=datetime(2026,1,1,0,20,tzinfo=UTC),
        reference_timestamp=datetime(2026,1,1,0,21,tzinfo=UTC),
        max_tick_age=pd.Timedelta(seconds=10),
    )
    assert report.misaligned_timeframe==2
    assert report.stale_tick
    assert {'misaligned_timeframe','stale_tick'}.issubset(report.failures)


def test_broker_spec_history_is_append_only_and_critical_change_degrades(tmp_path):
    path=tmp_path/'broker_specs.jsonl'
    monitor=BrokerSpecMonitor(path)
    now=datetime(2026,1,1,tzinfo=UTC)
    monitor.observe(spec(sessions=('MON:00:00-23:59',)),now)
    event=monitor.observe(spec(contract=1,sessions=('MON:00:00-23:59',)),now+timedelta(minutes=1))
    assert event and event.critical
    assert monitor.state('EURUSD') is OperationalState.DEGRADED
    monitor.manual_revalidate('EURUSD',now+timedelta(minutes=2))
    lines=[json.loads(line) for line in path.read_text().splitlines()]
    assert [x['record_type'] for x in lines]==['snapshot','snapshot','manual_revalidation']
    assert lines[0]['spec']['trading_sessions']==['MON:00:00-23:59']


def test_watcher_blocks_stale_tick_and_supports_volatility_percentile():
    config=WatcherConfig(enable_intrabar=True,stale_tick_seconds=5,volatility_percentile_threshold=0.95)
    blocked=should_evaluate(completed_bar=True,current_price=100,last_eval_price=99,atr=1,current_spread=1,rolling_spread_median=1,tick_age_seconds=6,config=config)
    assert not blocked.evaluate and blocked.blocked_reason=='stale_tick'
    vol=should_evaluate(completed_bar=False,current_price=100,last_eval_price=100,atr=1,current_spread=1,rolling_spread_median=1,volatility_percentile=0.97,config=config)
    assert vol.evaluate and 'volatility_percentile' in vol.reasons


def test_leakage_guard_blocks_embedded_and_suffixed_target_names():
    for column in ('rsi_target','forward_return_4','future_price','realized_r_after_costs'):
        with pytest.raises(LeakageError):
            assert_no_label_features(['rsi',column])
    assert_no_label_features(['rsi','atr_pct','ema_spread_pct'])


def test_frozen_holdout_detects_tampering_and_blocks_development_access(tmp_path):
    path=freeze(tmp_path)
    registry=HoldoutRegistry(path)
    with pytest.raises(RuntimeError,match='holdout leakage'):
        registry.assert_development_timestamp(datetime(2026,7,15,tzinfo=UTC))
    registry.assert_development_timestamp(datetime(2026,6,15,tzinfo=UTC))
    payload=json.loads(path.read_text())
    payload['holdout_end']='2026-09-01T00:00:00+00:00'
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError,match='integrity'):
        registry.load()
    with pytest.raises(RuntimeError,match='integrity'):
        TrainingGuard(path).assert_allowed()


def test_multiclass_boosters_expose_contract_classes_and_raw_margins(tmp_path):
    guard=TrainingGuard(freeze(tmp_path))
    X=np.arange(90,dtype=float).reshape(30,3)
    y=np.asarray([-1,0,1]*10)
    contract=n_bar_direction_contract(neutral_threshold=0.001)

    xgb=XGBoostModel(contract,n_estimators=5,max_depth=2).fit(X,y,guard=guard).predict_output(X[:4])
    assert np.array_equal(xgb.classes,np.asarray([-1,0,1]))
    assert xgb.probability.shape==(4,3)
    assert xgb.raw_score.shape==(4,3)

    lgb=LightGBMModel(contract,n_estimators=5,max_depth=2).fit(X,y,guard=guard).predict_output(X[:4])
    assert np.array_equal(lgb.classes,np.asarray([-1,0,1]))
    assert lgb.probability.shape==(4,3)
    assert lgb.raw_score.shape==(4,3)
