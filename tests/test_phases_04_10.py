from datetime import UTC, datetime
import pandas as pd
import pytest
from matamaple_trader.data.integrity import validate_bars
from matamaple_trader.broker.spec_integrity import BrokerSpecMonitor
from matamaple_trader.domain import *
from matamaple_trader.realtime.watcher import WatcherConfig, should_evaluate
from matamaple_trader.pipeline.core import SharedPipeline
from matamaple_trader.adapters.shared import HistoricalAdapter, MockLiveAdapter
from matamaple_trader.validation.leakage import LeakageError, assert_no_label_features, backward_asof_join
from matamaple_trader.validation.holdout import HoldoutPolicy, HoldoutRegistry, training_gate

def frame(n=3):
    t=pd.date_range('2026-01-01', periods=n, freq='15min', tz='UTC')
    return pd.DataFrame({'timestamp':t,'open':[1]*n,'high':[2]*n,'low':[0.5]*n,'close':[1.5]*n,'spread':[10]*n})
def test_integrity_happy_and_missing():
    assert validate_bars(frame(5),pd.Timedelta('15min'),min_bars=3,max_spread_points=20).ok
    f=frame(5).drop(index=2).reset_index(drop=True); assert 'missing_bars' in validate_bars(f,pd.Timedelta('15min'),min_bars=3).failures
def spec(contract=100000,tick=.00001,step=.01): return SymbolSpec('EURUSD',5,.00001,tick,1,contract,.01,100,step,-1,1)
def test_broker_critical_bypass():
    m=BrokerSpecMonitor(); now=datetime.now(UTC); assert m.observe(spec(),now) is None
    e=m.observe(spec(contract=1),now); assert e and e.critical and m.state('EURUSD')==OperationalState.DEGRADED
    m.manual_revalidate('EURUSD'); assert m.state('EURUSD')==OperationalState.ACTIVE
def test_watcher_measurable():
    d=should_evaluate(completed_bar=False,current_price=101,last_eval_price=100,atr=2,current_spread=1,rolling_spread_median=1,config=WatcherConfig(atr_move_multiple=.5,enable_intrabar=True)); assert d.evaluate and 'atr_move' in d.reasons
def market():
    t=datetime(2026,1,1,tzinfo=UTC); return MarketSnapshot(t,'EURUSD','M15',(Bar(t,1,2,.5,1.5),))
def test_shared_pipeline_adapter_consistency():
    p=SharedPipeline(); c=PipelineContext('v1.0.0'); assert HistoricalAdapter(p).evaluate(market(),c)==MockLiveAdapter(p).evaluate(market(),c)
def test_paused_is_avoid(): assert SharedPipeline().evaluate(market(),PipelineContext('v1',OperationalState.PAUSED)).signal==Signal.AVOID
def test_leakage_guards_and_backward_asof():
    with pytest.raises(LeakageError): assert_no_label_features(['rsi','target_return'])
    left=pd.DataFrame({'timestamp':pd.to_datetime(['2026-01-01 10:15Z'])}); right=pd.DataFrame({'timestamp':pd.to_datetime(['2026-01-01 10:00Z','2026-01-01 10:30Z']),'x':[1,2]}); assert backward_asof_join(left,right).iloc[0].x==1
def test_holdout_immutable_and_gate(tmp_path):
    p=HoldoutPolicy(datetime(2025,1,1,tzinfo=UTC),datetime(2025,6,1,tzinfo=UTC),datetime(2024,12,1,tzinfo=UTC),'v1'); r=HoldoutRegistry(tmp_path/'research_holdout.json'); r.freeze(p); assert r.load()['policy_version']=='v1'
    with pytest.raises(RuntimeError): r.freeze(p)
    with pytest.raises(RuntimeError): training_gate(shared_pipeline_exists=True, leakage_framework_exists=True, holdout_frozen=False)
    training_gate(shared_pipeline_exists=True, leakage_framework_exists=True, holdout_frozen=True)
