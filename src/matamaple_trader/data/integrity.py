from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True, slots=True)
class DataQualityReport:
    ok: bool; duplicate_timestamps:int; missing_ranges:tuple[str,...]; impossible_ohlc:int; time_reversal:bool; abnormal_spread:int; insufficient_history:bool
    @property
    def failures(self)->tuple[str,...]:
        out=[]
        if self.duplicate_timestamps: out.append('duplicate_timestamps')
        if self.missing_ranges: out.append('missing_bars')
        if self.impossible_ohlc: out.append('impossible_ohlc')
        if self.time_reversal: out.append('time_reversal')
        if self.abnormal_spread: out.append('abnormal_spread')
        if self.insufficient_history: out.append('insufficient_history')
        return tuple(out)

def validate_bars(frame:pd.DataFrame, expected_delta:pd.Timedelta|None=None, min_bars:int=100, max_spread_points:int|None=None)->DataQualityReport:
    if frame.empty: return DataQualityReport(False,0,(),0,False,0,True)
    ts=pd.to_datetime(frame['timestamp'],utc=True)
    dup=int(ts.duplicated().sum()); reversal=not ts.is_monotonic_increasing
    impossible=int(((frame['high'] < frame[['open','close','low']].max(axis=1)) | (frame['low'] > frame[['open','close','high']].min(axis=1)) | (frame['low']>frame['high'])).sum())
    missing=[]
    if expected_delta is not None and len(ts)>1:
        diffs=ts.diff()
        for i in diffs[diffs>expected_delta].index: missing.append(f'{ts.iloc[i-1].isoformat()}->{ts.iloc[i].isoformat()}')
    abnormal=0
    if max_spread_points is not None and 'spread' in frame: abnormal=int((frame['spread']>max_spread_points).sum())
    insufficient=len(frame)<min_bars
    ok=not (dup or missing or impossible or reversal or abnormal or insufficient)
    return DataQualityReport(ok,dup,tuple(missing),impossible,reversal,abnormal,insufficient)
