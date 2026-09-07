from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

@dataclass(frozen=True, slots=True)
class HoldoutPolicy:
    holdout_start: datetime; holdout_end: datetime; freeze_timestamp: datetime; policy_version: str; kind: str='research_holdout'; cycle_id: str|None=None
    def __post_init__(self):
        for v in (self.holdout_start,self.holdout_end,self.freeze_timestamp):
            if v.tzinfo is None: raise ValueError('holdout timestamps must be timezone-aware')
        if self.holdout_start>=self.holdout_end: raise ValueError('holdout_start must precede holdout_end')

class HoldoutRegistry:
    def __init__(self,path:str|Path): self.path=Path(path)
    def freeze(self,policy:HoldoutPolicy)->None:
        if self.path.exists(): raise RuntimeError('holdout policy already frozen; never overwrite it')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        payload=asdict(policy)
        for k in ('holdout_start','holdout_end','freeze_timestamp'): payload[k]=payload[k].astimezone(UTC).isoformat()
        self.path.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    def load(self)->dict: return json.loads(self.path.read_text(encoding='utf-8'))

def training_gate(*,shared_pipeline_exists:bool,leakage_framework_exists:bool,holdout_frozen:bool)->None:
    missing=[name for name,ok in [('shared_pipeline',shared_pipeline_exists),('leakage_framework',leakage_framework_exists),('frozen_holdout',holdout_frozen)] if not ok]
    if missing: raise RuntimeError('training blocked; missing critical build gate: '+', '.join(missing))
