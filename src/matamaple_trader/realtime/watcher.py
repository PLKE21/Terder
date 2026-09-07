from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class WatcherConfig:
    atr_move_multiple: float=0.5; spread_spike_multiplier: float=2.0; enable_intrabar: bool=False
@dataclass(frozen=True, slots=True)
class TriggerDecision:
    evaluate: bool; reasons: tuple[str,...]

def should_evaluate(*,completed_bar:bool,current_price:float,last_eval_price:float|None,atr:float|None,current_spread:float|None,rolling_spread_median:float|None,structure_break:bool=False,config:WatcherConfig=WatcherConfig())->TriggerDecision:
    reasons=[]
    if completed_bar: reasons.append('completed_bar')
    if config.enable_intrabar:
        if last_eval_price is not None and atr and abs(current_price-last_eval_price)>=config.atr_move_multiple*atr: reasons.append('atr_move')
        if current_spread is not None and rolling_spread_median and current_spread>rolling_spread_median*config.spread_spike_multiplier: reasons.append('spread_spike')
        if structure_break: reasons.append('structure_break')
    return TriggerDecision(bool(reasons),tuple(reasons))
