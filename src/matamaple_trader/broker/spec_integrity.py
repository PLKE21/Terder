from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from matamaple_trader.domain import OperationalState, SymbolSpec

CRITICAL_FIELDS=('contract_size','tick_size','volume_step')

@dataclass(frozen=True, slots=True)
class BrokerSpecEvent:
    timestamp: datetime
    symbol: str
    changed_fields: tuple[str,...]
    critical: bool
    new_state: OperationalState

class BrokerSpecMonitor:
    def __init__(self, history_path: str|Path|None=None):
        self._last:dict[str,SymbolSpec]={}
        self._state:dict[str,OperationalState]={}
        self.history_path=Path(history_path) if history_path is not None else None

    def state(self,symbol:str)->OperationalState:
        return self._state.get(symbol,OperationalState.ACTIVE)

    def _append(self, payload:dict)->None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True,exist_ok=True)
        with self.history_path.open('a',encoding='utf-8') as fh:
            fh.write(json.dumps(payload,sort_keys=True)+"\n")

    def observe(self,spec:SymbolSpec,timestamp:datetime)->BrokerSpecEvent|None:
        if timestamp.tzinfo is None:
            raise ValueError('broker spec timestamp must be timezone-aware')
        timestamp=timestamp.astimezone(UTC)
        prev=self._last.get(spec.symbol)
        self._last[spec.symbol]=spec

        changed:tuple[str,...]=()
        critical=False
        event=None
        if prev is None:
            self._state.setdefault(spec.symbol,OperationalState.ACTIVE)
        else:
            a,b=asdict(prev),asdict(spec)
            changed=tuple(k for k in a if k!='symbol' and a[k]!=b[k])
            if changed:
                critical=any(k in CRITICAL_FIELDS for k in changed)
                if critical:
                    self._state[spec.symbol]=OperationalState.DEGRADED
                event=BrokerSpecEvent(timestamp,spec.symbol,changed,critical,self.state(spec.symbol))

        self._append({
            'record_type':'snapshot',
            'timestamp':timestamp.isoformat(),
            'symbol':spec.symbol,
            'state':self.state(spec.symbol).value,
            'changed_fields':list(changed),
            'critical':critical,
            'spec':asdict(spec),
        })
        return event

    def manual_revalidate(self,symbol:str,timestamp:datetime|None=None)->None:
        if symbol not in self._last:
            raise KeyError(f'no broker spec snapshot for {symbol}')
        when=(timestamp or datetime.now(UTC))
        if when.tzinfo is None:
            raise ValueError('manual revalidation timestamp must be timezone-aware')
        self._state[symbol]=OperationalState.ACTIVE
        self._append({
            'record_type':'manual_revalidation',
            'timestamp':when.astimezone(UTC).isoformat(),
            'symbol':symbol,
            'state':OperationalState.ACTIVE.value,
        })
