from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True, slots=True)
class DataQualityReport:
    ok: bool
    duplicate_timestamps:int
    missing_ranges:tuple[str,...]
    impossible_ohlc:int
    time_reversal:bool
    abnormal_spread:int
    insufficient_history:bool
    stale_tick:bool=False
    misaligned_timeframe:int=0

    @property
    def failures(self)->tuple[str,...]:
        out=[]
        if self.duplicate_timestamps: out.append('duplicate_timestamps')
        if self.missing_ranges: out.append('missing_bars')
        if self.impossible_ohlc: out.append('impossible_ohlc')
        if self.time_reversal: out.append('time_reversal')
        if self.abnormal_spread: out.append('abnormal_spread')
        if self.insufficient_history: out.append('insufficient_history')
        if self.stale_tick: out.append('stale_tick')
        if self.misaligned_timeframe: out.append('misaligned_timeframe')
        return tuple(out)

def validate_bars(
    frame:pd.DataFrame,
    expected_delta:pd.Timedelta|None=None,
    min_bars:int=100,
    max_spread_points:int|None=None,
    *,
    latest_tick_timestamp=None,
    reference_timestamp=None,
    max_tick_age:pd.Timedelta|None=None,
    alignment_origin=None,
)->DataQualityReport:
    if frame.empty:
        return DataQualityReport(False,0,(),0,False,0,True)

    required={'timestamp','open','high','low','close'}
    missing_cols=required.difference(frame.columns)
    if missing_cols:
        raise ValueError(f"missing required columns: {sorted(missing_cols)}")

    ts=pd.to_datetime(frame['timestamp'],utc=True)
    dup=int(ts.duplicated().sum())
    reversal=not ts.is_monotonic_increasing

    impossible=int(((frame['high'] < frame[['open','close','low']].max(axis=1)) | (frame['low'] > frame[['open','close','high']].min(axis=1)) | (frame['low']>frame['high'])).sum())

    missing=[]
    misaligned=0
    if expected_delta is not None:
        if expected_delta <= pd.Timedelta(0):
            raise ValueError('expected_delta must be positive')
        if len(ts)>1:
            diffs=ts.diff()
            for i in diffs[diffs>expected_delta].index:
                missing.append(f'{ts.iloc[i-1].isoformat()}->{ts.iloc[i].isoformat()}')
        origin=pd.Timestamp('1970-01-01T00:00:00Z') if alignment_origin is None else pd.Timestamp(alignment_origin)
        if origin.tzinfo is None:
            origin=origin.tz_localize('UTC')
        else:
            origin=origin.tz_convert('UTC')
        misaligned=int((((ts-origin) % expected_delta) != pd.Timedelta(0)).sum())

    abnormal=0
    spread_col='spread' if 'spread' in frame else ('spread_points' if 'spread_points' in frame else None)
    if max_spread_points is not None and spread_col is not None:
        abnormal=int((pd.to_numeric(frame[spread_col],errors='coerce')>max_spread_points).sum())

    stale=False
    if latest_tick_timestamp is not None and reference_timestamp is not None and max_tick_age is not None:
        tick_ts=pd.Timestamp(latest_tick_timestamp)
        ref_ts=pd.Timestamp(reference_timestamp)
        tick_ts=tick_ts.tz_localize('UTC') if tick_ts.tzinfo is None else tick_ts.tz_convert('UTC')
        ref_ts=ref_ts.tz_localize('UTC') if ref_ts.tzinfo is None else ref_ts.tz_convert('UTC')
        stale=(ref_ts-tick_ts)>max_tick_age

    insufficient=len(frame)<min_bars
    ok=not (dup or missing or impossible or reversal or abnormal or insufficient or stale or misaligned)
    return DataQualityReport(ok,dup,tuple(missing),impossible,reversal,abnormal,insufficient,bool(stale),misaligned)
