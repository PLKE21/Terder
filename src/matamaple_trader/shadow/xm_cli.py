from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import sys
import time

import pandas as pd

from matamaple_trader.adapters.mt5_connector import MT5Connector
from matamaple_trader.domain import Bar, MarketSnapshot, OperationalState, PipelineContext
from matamaple_trader.pipeline.locked_runtime import LockedQuantPipeline
from matamaple_trader.validation.candidate_lock import CandidateLockRegistry
from .runner import ShadowRunner
from .store import ShadowStore


TIMEFRAME_DELTAS={
    'M1':pd.Timedelta('1min'),'M5':pd.Timedelta('5min'),'M15':pd.Timedelta('15min'),
    'M30':pd.Timedelta('30min'),'H1':pd.Timedelta('1h'),'H4':pd.Timedelta('4h'),'D1':pd.Timedelta('1d'),
}


def _completed(frame: pd.DataFrame, delta: pd.Timedelta, now: datetime) -> pd.DataFrame:
    if frame.empty:
        return frame
    out=frame.copy()
    out['timestamp']=pd.to_datetime(out['timestamp'],utc=True)
    return out.loc[out['timestamp']+delta <= pd.Timestamp(now)].sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)


def _bars(frame: pd.DataFrame) -> tuple[Bar,...]:
    return tuple(Bar(
        timestamp=row.timestamp.to_pydatetime(), open=float(row.open), high=float(row.high), low=float(row.low), close=float(row.close),
        tick_volume=int(getattr(row,'tick_volume',0)), spread_points=int(getattr(row,'spread',0)),
    ) for row in frame.itertuples(index=False))


def _attach_ready_outcomes(store: ShadowStore, completed: pd.DataFrame, *, horizon_bars: int) -> int:
    if completed.empty:
        return 0
    times=list(pd.to_datetime(completed['timestamp'],utc=True))
    index={ts:i for i,ts in enumerate(times)}
    attached=0
    for pending in store.pending_outcomes():
        obs=pending['payload'].get('observation',{})
        if not isinstance(obs,dict) or 'reference_bar_timestamp' not in obs or 'reference_close' not in obs:
            continue
        ref=pd.Timestamp(obs['reference_bar_timestamp'])
        if ref.tzinfo is None: ref=ref.tz_localize('UTC')
        else: ref=ref.tz_convert('UTC')
        pos=index.get(ref)
        if pos is None or pos+horizon_bars>=len(completed):
            continue
        future=completed.iloc[pos+horizon_bars]
        reference_close=float(obs['reference_close']); outcome_close=float(future['close'])
        ret=outcome_close/reference_close-1.0
        store.attach_outcome(int(pending['prediction_id']),{
            'horizon_bars':horizon_bars,'reference_close':reference_close,'outcome_close':outcome_close,
            'forward_return':ret,'label':int(ret>0),'outcome_bar_timestamp':pd.Timestamp(future['timestamp']).isoformat(),
            'attached_at':datetime.now(UTC).isoformat(),
        })
        attached+=1
    return attached


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Read-only XM shadow runner using the locked shared quantitative pipeline. Never sends orders.')
    p.add_argument('--symbol',default='EURUSD')
    p.add_argument('--timeframe',choices=tuple(TIMEFRAME_DELTAS),default='H1')
    p.add_argument('--parquet',required=True,help='Approved development source parquet used by the locked candidate.')
    p.add_argument('--holdout-registry',default='data/validation/frozen_holdout.json')
    p.add_argument('--candidate-lock',default='data/validation/candidate_lock.json')
    p.add_argument('--db',default='data/shadow/shadow.db')
    p.add_argument('--lookback-bars',type=int,default=500)
    p.add_argument('--poll-seconds',type=float,default=5.0)
    p.add_argument('--once',action='store_true',help='Evaluate at most the latest newly closed bar and exit.')
    args=p.parse_args(argv)
    if args.lookback_bars < 50 or args.poll_seconds <= 0: p.error('lookback-bars must be >=50 and poll-seconds >0')

    try: import MetaTrader5 as mt5
    except ImportError as exc:
        p.error("MetaTrader5 is not installed. Install with pip install -e '.[mt5]'"); raise exc

    lock=CandidateLockRegistry(args.candidate_lock).load()
    pipeline=LockedQuantPipeline.from_artifacts(parquet_path=args.parquet,holdout_registry_path=args.holdout_registry,candidate_lock_path=args.candidate_lock)
    store=ShadowStore(args.db); runner=ShadowRunner(pipeline,store)
    connector=MT5Connector(mt5); delta=TIMEFRAME_DELTAS[args.timeframe]; timeframe_code=getattr(mt5,f'TIMEFRAME_{args.timeframe}')
    connector.connect(); last_seen=None
    try:
        while True:
            now=datetime.now(UTC)
            # Calendar lookback is intentionally generous; completed bars, not wall-clock gaps, determine the horizon outcome.
            start=now-timedelta(seconds=float(delta.total_seconds())*max(args.lookback_bars*2,200))
            frame=_completed(connector.get_rates(args.symbol,timeframe_code,start,now),delta,now)
            if len(frame)<50: raise RuntimeError('insufficient completed bars for shadow feature generation')
            _attach_ready_outcomes(store,frame,horizon_bars=int(lock['horizon_bars']))
            latest_open=pd.Timestamp(frame.iloc[-1]['timestamp'])
            latest_close=(latest_open+delta).to_pydatetime()
            if last_seen is None:
                existing={x.timestamp for x in store.list_predictions() if x.symbol==args.symbol and x.pipeline_version==lock['pipeline_version']}
                last_seen=latest_close.isoformat() if latest_close.isoformat() in existing else None
            if last_seen != latest_close.isoformat():
                spec=connector.get_symbol_spec(args.symbol); tick=connector.get_tick(args.symbol)
                market=MarketSnapshot(timestamp=latest_close,symbol=args.symbol,timeframe=args.timeframe,bars=_bars(frame),tick=tick,metadata={'point':spec.point})
                context=PipelineContext(pipeline_version=lock['pipeline_version'],operational_state=OperationalState.ACTIVE,source='xm_shadow')
                result,prediction_id=runner.evaluate_and_store(market,context,observation={
                    'reference_bar_timestamp':latest_open.isoformat(),'reference_close':float(frame.iloc[-1]['close']),
                    'horizon_bars':int(lock['horizon_bars']),'timeframe':args.timeframe,'candidate_id':lock['candidate_id'],
                })
                last_seen=latest_close.isoformat()
                print(json.dumps({'prediction_id':prediction_id,'timestamp':result.timestamp.isoformat(),'symbol':result.symbol,'signal':result.signal.value,'probability':result.calibrated_probability,'grade':result.grade,'regime':result.regime,'stats':store.stats()},sort_keys=True),flush=True)
            if args.once: break
            time.sleep(args.poll_seconds)
    finally:
        connector.close()
    return 0


if __name__=='__main__': sys.exit(main())
