from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd

@dataclass(frozen=True, slots=True)
class CollectionMetadata:
    symbol: str
    timeframe: str
    first_available_timestamp: datetime|None
    last_available_timestamp: datetime|None
    bar_count: int
    source: str
    missing_ranges: tuple[str,...]=()

class HistoricalCollector:
    def __init__(self, adapter, root: str|Path='data/historical') -> None:
        self.adapter=adapter
        self.root=Path(root)

    def collect(
        self,
        symbol:str,
        timeframe_code:int,
        timeframe_name:str,
        start_utc:datetime,
        end_utc:datetime,
        *,
        expected_delta:pd.Timedelta|None=None,
    )->tuple[pd.DataFrame,CollectionMetadata]:
        frame=self.adapter.get_rates(symbol,timeframe_code,start_utc,end_utc).copy()
        if not frame.empty:
            frame['timestamp']=pd.to_datetime(frame['timestamp'], utc=True)

        # Important: do not sort or deduplicate here. Integrity validation must see
        # the raw ordering/duplicates returned by the source so defects cannot be hidden.
        first_ts=None if frame.empty else frame['timestamp'].min().to_pydatetime()
        last_ts=None if frame.empty else frame['timestamp'].max().to_pydatetime()

        missing: list[str]=[]
        if expected_delta is not None and not frame.empty:
            ordered=frame[['timestamp']].drop_duplicates().sort_values('timestamp').reset_index(drop=True)
            diffs=ordered['timestamp'].diff()
            for i in diffs[diffs>expected_delta].index:
                missing.append(f"{ordered['timestamp'].iloc[i-1].isoformat()}->{ordered['timestamp'].iloc[i].isoformat()}")

        meta=CollectionMetadata(
            symbol=symbol,
            timeframe=timeframe_name,
            first_available_timestamp=first_ts,
            last_available_timestamp=last_ts,
            bar_count=len(frame),
            source=self.adapter.__class__.__name__,
            missing_ranges=tuple(missing),
        )
        return frame, meta

    def write_parquet(self, frame:pd.DataFrame, symbol:str, timeframe:str)->Path:
        path=self.root/symbol/f'{timeframe}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path,index=False)
        return path
