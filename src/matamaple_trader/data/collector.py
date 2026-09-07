from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd

@dataclass(frozen=True, slots=True)
class CollectionMetadata:
    symbol: str; timeframe: str; first_available_timestamp: datetime|None; last_available_timestamp: datetime|None; bar_count: int; source: str

class HistoricalCollector:
    def __init__(self, adapter, root: str|Path='data/historical') -> None:
        self.adapter=adapter; self.root=Path(root)
    def collect(self, symbol:str, timeframe_code:int, timeframe_name:str, start_utc:datetime, end_utc:datetime)->tuple[pd.DataFrame,CollectionMetadata]:
        frame=self.adapter.get_rates(symbol,timeframe_code,start_utc,end_utc).copy()
        if not frame.empty:
            frame['timestamp']=pd.to_datetime(frame['timestamp'], utc=True)
            frame=frame.sort_values('timestamp').drop_duplicates('timestamp', keep='last').reset_index(drop=True)
        meta=CollectionMetadata(symbol,timeframe_name,None if frame.empty else frame.timestamp.iloc[0].to_pydatetime(),None if frame.empty else frame.timestamp.iloc[-1].to_pydatetime(),len(frame),self.adapter.__class__.__name__)
        return frame, meta
    def write_parquet(self, frame:pd.DataFrame, symbol:str, timeframe:str)->Path:
        path=self.root/symbol/f'{timeframe}.parquet'; path.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(path,index=False); return path
