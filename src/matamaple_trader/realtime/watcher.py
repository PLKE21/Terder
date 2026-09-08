from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class WatcherConfig:
    atr_move_multiple: float=0.5
    spread_spike_multiplier: float=2.0
    volatility_percentile_threshold: float=0.95
    stale_tick_seconds: float=10.0
    enable_intrabar: bool=False
    watcher_version: str='watcher-v1'

    def __post_init__(self)->None:
        if self.atr_move_multiple<=0 or self.spread_spike_multiplier<=0 or self.stale_tick_seconds<=0:
            raise ValueError('watcher thresholds must be positive')
        if not 0 < self.volatility_percentile_threshold <= 1:
            raise ValueError('volatility_percentile_threshold must be in (0, 1]')

@dataclass(frozen=True, slots=True)
class TriggerDecision:
    evaluate: bool
    reasons: tuple[str,...]
    blocked_reason: str|None=None

def should_evaluate(
    *,
    completed_bar:bool,
    current_price:float,
    last_eval_price:float|None,
    atr:float|None,
    current_spread:float|None,
    rolling_spread_median:float|None,
    structure_break:bool=False,
    volatility_percentile:float|None=None,
    tick_age_seconds:float|None=None,
    config:WatcherConfig=WatcherConfig(),
)->TriggerDecision:
    if tick_age_seconds is not None and tick_age_seconds>config.stale_tick_seconds:
        return TriggerDecision(False,(),blocked_reason='stale_tick')

    reasons=[]
    if completed_bar:
        reasons.append('completed_bar')
    if config.enable_intrabar:
        if last_eval_price is not None and atr is not None and atr>0 and abs(current_price-last_eval_price)>=config.atr_move_multiple*atr:
            reasons.append('atr_move')
        if current_spread is not None and rolling_spread_median is not None and rolling_spread_median>0 and current_spread>rolling_spread_median*config.spread_spike_multiplier:
            reasons.append('spread_spike')
        if structure_break:
            reasons.append('structure_break')
        if volatility_percentile is not None and volatility_percentile>=config.volatility_percentile_threshold:
            reasons.append('volatility_percentile')
    return TriggerDecision(bool(reasons),tuple(reasons))
