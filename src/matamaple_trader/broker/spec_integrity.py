from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from matamaple_trader.domain import OperationalState, SymbolSpec

CRITICAL_FIELDS=('contract_size','tick_size','volume_step')
@dataclass(frozen=True, slots=True)
class BrokerSpecEvent:
    timestamp: datetime; symbol: str; changed_fields: tuple[str,...]; critical: bool; new_state: OperationalState

class BrokerSpecMonitor:
    def __init__(self): self._last:dict[str,SymbolSpec]={}; self._state:dict[str,OperationalState]={}
    def state(self,symbol:str)->OperationalState: return self._state.get(symbol,OperationalState.ACTIVE)
    def observe(self,spec:SymbolSpec,timestamp:datetime)->BrokerSpecEvent|None:
        prev=self._last.get(spec.symbol); self._last[spec.symbol]=spec
        if prev is None: self._state.setdefault(spec.symbol,OperationalState.ACTIVE); return None
        a,b=asdict(prev),asdict(spec); changed=tuple(k for k in a if k!='symbol' and a[k]!=b[k])
        if not changed: return None
        critical=any(k in CRITICAL_FIELDS for k in changed)
        if critical: self._state[spec.symbol]=OperationalState.DEGRADED
        return BrokerSpecEvent(timestamp,spec.symbol,changed,critical,self.state(spec.symbol))
    def manual_revalidate(self,symbol:str)->None: self._state[symbol]=OperationalState.ACTIVE
