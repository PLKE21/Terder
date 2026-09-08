from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import sys
import pandas as pd

from matamaple_trader.adapters.mt5_connector import MT5Connector
from .fbs_smoke import run_readiness_check

TIMEFRAME_DELTAS={
    'M1':pd.Timedelta('1min'),
    'M5':pd.Timedelta('5min'),
    'M15':pd.Timedelta('15min'),
    'M30':pd.Timedelta('30min'),
    'H1':pd.Timedelta('1h'),
    'H4':pd.Timedelta('4h'),
    'D1':pd.Timedelta('1d'),
}


def _parse_utc(value:str)->datetime:
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError('timestamp must include timezone, e.g. 2026-09-01T00:00:00+00:00')
    return dt.astimezone(UTC)


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description='Read-only FBS/MT5 readiness smoke check. Never sends orders.')
    parser.add_argument('--symbol',default='EURUSD')
    parser.add_argument('--timeframe',choices=tuple(TIMEFRAME_DELTAS),default='M15')
    parser.add_argument('--start',required=True,type=_parse_utc)
    parser.add_argument('--end',required=True,type=_parse_utc)
    parser.add_argument('--min-bars',type=int,default=100)
    parser.add_argument('--max-spread-points',type=int,default=None)
    parser.add_argument('--max-tick-age-seconds',type=float,default=10.0)
    parser.add_argument('--check-continuity',action='store_true',help='Enable exact bar-gap checks only after validating the FBS symbol/session calendar for the requested period.')
    parser.add_argument('--broker-history',default='data/fbs/broker/spec_history.jsonl')
    args=parser.parse_args(argv)

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        parser.error("MetaTrader5 is not installed. Install with pip install -e '.[mt5]'")
        raise exc

    timeframe_code=getattr(mt5,f'TIMEFRAME_{args.timeframe}')
    connector=MT5Connector(mt5)
    try:
        connector.connect()
        report=run_readiness_check(
            connector,
            symbol=args.symbol,
            timeframe_code=timeframe_code,
            start_utc=args.start,
            end_utc=args.end,
            expected_delta=TIMEFRAME_DELTAS[args.timeframe] if args.check_continuity else None,
            min_bars=args.min_bars,
            max_spread_points=args.max_spread_points,
            max_tick_age_seconds=args.max_tick_age_seconds,
            broker_history_path=args.broker_history,
        )
    finally:
        connector.close()

    payload=asdict(report)
    payload['checked_at']=report.checked_at.isoformat()
    payload['broker_state']=report.broker_state.value
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0 if report.ready_for_collection else 2


if __name__=='__main__':
    sys.exit(main())
