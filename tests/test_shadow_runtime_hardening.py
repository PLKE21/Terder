from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from matamaple_trader.domain import OperationalState, Signal, SignalResult
from matamaple_trader.shadow.store import ShadowStore
from matamaple_trader.shadow.xm_cli import _attach_ready_outcomes, _completed


def _result(ts: datetime) -> SignalResult:
    return SignalResult(ts,'EURUSD',Signal.WAIT,0.1,0.55,'TREND',None,None,None,None,None,None,None,'C','NORMAL',None,'v0.1.0',OperationalState.ACTIVE)


def test_shadow_observation_is_immutable_and_outcome_uses_bar_horizon(tmp_path):
    store=ShadowStore(tmp_path/'shadow.db')
    base=datetime(2026,9,1,0,tzinfo=UTC)
    pid=store.append_prediction(_result(base+timedelta(hours=1)),observation={
        'reference_bar_timestamp':base.isoformat(),'reference_close':1.1000,'horizon_bars':4,
    })
    frame=pd.DataFrame({
        'timestamp':[base+timedelta(hours=i) for i in range(6)],
        'open':[1.1]*6,'high':[1.2]*6,'low':[1.0]*6,
        'close':[1.1000,1.1010,1.1020,1.1030,1.1040,1.1050],
    })
    assert _attach_ready_outcomes(store,frame,horizon_bars=4)==1
    assert store.stats()=={'predictions':1,'outcomes':1,'outcome_coverage':1.0}
    assert store.pending_outcomes()==()
    try:
        store.attach_outcome(pid,{'label':0})
        assert False,'outcome overwrite must fail'
    except RuntimeError:
        pass


def test_completed_excludes_current_open_bar():
    base=datetime(2026,9,1,0,tzinfo=UTC)
    frame=pd.DataFrame({'timestamp':[base,base+timedelta(hours=1)],'close':[1.0,2.0]})
    out=_completed(frame,pd.Timedelta('1h'),base+timedelta(hours=1,minutes=30))
    assert len(out)==1
    assert pd.Timestamp(out.iloc[0]['timestamp'])==pd.Timestamp(base)
