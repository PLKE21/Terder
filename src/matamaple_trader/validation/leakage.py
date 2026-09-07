from __future__ import annotations
import pandas as pd

class LeakageError(RuntimeError): pass
LABEL_PREFIXES=('label','target','future_')
def assert_no_label_features(columns)->None:
    bad=[c for c in columns if str(c).lower().startswith(LABEL_PREFIXES)]
    if bad: raise LeakageError(f'label-derived feature columns forbidden: {bad}')
def assert_monotonic_time(frame:pd.DataFrame)->None:
    ts=pd.to_datetime(frame['timestamp'],utc=True)
    if not ts.is_monotonic_increasing or ts.duplicated().any(): raise LeakageError('timestamps must be strictly chronological and unique')
def backward_asof_join(left:pd.DataFrame,right:pd.DataFrame,on:str='timestamp')->pd.DataFrame:
    l=left.sort_values(on).copy(); r=right.sort_values(on).copy()
    return pd.merge_asof(l,r,on=on,direction='backward',allow_exact_matches=True)
