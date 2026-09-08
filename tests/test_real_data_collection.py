from datetime import UTC, datetime, timedelta

import pandas as pd

from matamaple_trader.data.xm_collection import CollectionRequest, XMHistoricalCollectionPipeline


class GoodConnector:
    def __init__(self):
        self.calls=[]

    def get_rates(self,symbol,timeframe,start,end):
        self.calls.append((start,end))
        # One bar at each logical chunk start; raw source remains ordered and unique.
        return pd.DataFrame({
            'timestamp':pd.to_datetime([start],utc=True),
            'open':[1.0], 'high':[2.0], 'low':[0.5], 'close':[1.5], 'spread':[5],
        })


class BadConnector:
    def get_rates(self,symbol,timeframe,start,end):
        t=pd.Timestamp(start)
        return pd.DataFrame({
            'timestamp':[t,t],
            'open':[1.0,1.0], 'high':[2.0,2.0], 'low':[0.5,0.5], 'close':[1.5,1.5],
        })


def request(*,days=3,chunk_days=1,min_bars=1):
    start=datetime(2026,1,1,tzinfo=UTC)
    return CollectionRequest(
        symbol='EURUSD', timeframe_code=15, timeframe_name='M15',
        start_utc=start, end_utc=start+timedelta(days=days),
        expected_delta=None, min_bars=min_bars, chunk_days=chunk_days,
    )


def test_chunk_queries_do_not_overlap_and_good_data_writes(tmp_path,monkeypatch):
    connector=GoodConnector()
    monkeypatch.setattr(pd.DataFrame,'to_parquet',lambda self,path,index=False: path.write_text('parquet-placeholder'))
    frame,report,manifest=XMHistoricalCollectionPipeline(connector,tmp_path).collect(request())
    assert report.ok and manifest.quality_ok
    assert manifest.parquet_path is not None and manifest.dataset_sha256 is not None
    assert len(frame)==3 and len(connector.calls)==3
    for (_,left_end),(right_start,_) in zip(connector.calls,connector.calls[1:]):
        assert left_end < right_start


def test_bad_raw_data_never_writes_parquet(tmp_path):
    _,report,manifest=XMHistoricalCollectionPipeline(BadConnector(),tmp_path).collect(request(days=1))
    assert not report.ok
    assert 'duplicate_timestamps' in report.failures
    assert manifest.parquet_path is None and manifest.dataset_sha256 is None
    assert not (tmp_path/'EURUSD'/'M15.parquet').exists()
    assert (tmp_path/'EURUSD'/'M15.manifest.json').exists()
