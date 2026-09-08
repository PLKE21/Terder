from __future__ import annotations
import re
import pandas as pd

class LeakageError(RuntimeError):
    pass

FORBIDDEN_FEATURE_TOKENS={'label','target','future','outcome'}
FORBIDDEN_COMPACT_MARKERS=('forwardreturn','realizedreturn','realizedr')

def _normalized_tokens(name:object)->tuple[set[str],str]:
    text=str(name).strip().lower()
    compact=re.sub(r'[^a-z0-9]+','',text)
    tokens=set(filter(None,re.split(r'[^a-z0-9]+',text)))
    return tokens,compact

def assert_no_label_features(columns)->None:
    bad=[]
    for column in columns:
        tokens,compact=_normalized_tokens(column)
        if tokens & FORBIDDEN_FEATURE_TOKENS or any(marker in compact for marker in FORBIDDEN_COMPACT_MARKERS):
            bad.append(column)
    if bad:
        raise LeakageError(f'label/future-derived feature columns forbidden: {bad}')

def assert_monotonic_time(frame:pd.DataFrame)->None:
    if 'timestamp' not in frame:
        raise LeakageError('timestamp column is required')
    ts=pd.to_datetime(frame['timestamp'],utc=True)
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise LeakageError('timestamps must be strictly chronological and unique')

def backward_asof_join(left:pd.DataFrame,right:pd.DataFrame,on:str='timestamp')->pd.DataFrame:
    if on not in left or on not in right:
        raise LeakageError(f'{on} must exist in both frames')
    l=left.copy(); r=right.copy()
    l[on]=pd.to_datetime(l[on],utc=True); r[on]=pd.to_datetime(r[on],utc=True)
    assert_monotonic_time(l.rename(columns={on:'timestamp'}) if on!='timestamp' else l)
    assert_monotonic_time(r.rename(columns={on:'timestamp'}) if on!='timestamp' else r)
    return pd.merge_asof(l,r,on=on,direction='backward',allow_exact_matches=True)
