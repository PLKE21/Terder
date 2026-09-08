from datetime import UTC, datetime, timedelta

import pandas as pd

from matamaple_trader.domain import SymbolSpec, Tick
from matamaple_trader.readiness import run_readiness_check


class FakeConnector:
    def __init__(self, tick_age_seconds=1):
        self.now=datetime(2026,9,8,12,0,tzinfo=UTC)
        self.tick_age_seconds=tick_age_seconds

    def get_tick(self,symbol):
        return Tick(self.now-timedelta(seconds=self.tick_age_seconds),1.1000,1.1001)

    def get_symbol_spec(self,symbol):
        return SymbolSpec(symbol,5,0.00001,0.00001,1.0,100000,0.01,100.0,0.01,-1.0,1.0)

    def get_rates(self,symbol,timeframe,start,end):
        ts=pd.date_range('2026-09-08T10:00:00Z',periods=5,freq='15min')
        return pd.DataFrame({
            'timestamp':ts,
            'open':[1.1]*5,
            'high':[1.2]*5,
            'low':[1.0]*5,
            'close':[1.15]*5,
            'spread':[10]*5,
        })


def test_readiness_check_allows_clean_read_only_market_data(tmp_path):
    connector=FakeConnector(1)
    report=run_readiness_check(
        connector,
        symbol='EURUSD',
        timeframe_code=15,
        start_utc=datetime(2026,9,8,10,0,tzinfo=UTC),
        end_utc=datetime(2026,9,8,11,0,tzinfo=UTC),
        expected_delta=pd.Timedelta('15min'),
        min_bars=5,
        max_spread_points=20,
        broker_history_path=tmp_path/'broker.jsonl',
        now_utc=connector.now,
    )
    assert report.ready_for_collection
    assert report.quality_failures==()
    assert report.continuity_checked
    assert (tmp_path/'broker.jsonl').exists()


def test_readiness_check_fails_safe_on_stale_tick():
    connector=FakeConnector(30)
    report=run_readiness_check(
        connector,
        symbol='EURUSD',
        timeframe_code=15,
        start_utc=datetime(2026,9,8,10,0,tzinfo=UTC),
        end_utc=datetime(2026,9,8,11,0,tzinfo=UTC),
        expected_delta=pd.Timedelta('15min'),
        min_bars=5,
        max_tick_age_seconds=5,
        now_utc=connector.now,
    )
    assert not report.ready_for_collection
    assert 'stale_tick' in report.quality_failures
