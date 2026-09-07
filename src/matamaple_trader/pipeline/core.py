from __future__ import annotations
from typing import Protocol
from matamaple_trader.domain import MarketSnapshot, PipelineContext, SignalResult, Signal, OperationalState

class PipelineStage(Protocol):
    def evaluate(self, market:MarketSnapshot, context:PipelineContext)->SignalResult: ...

class SharedPipeline:
    def evaluate(self, market:MarketSnapshot, context:PipelineContext)->SignalResult:
        state=context.operational_state
        signal=Signal.AVOID if state is OperationalState.PAUSED else Signal.WAIT
        return SignalResult(timestamp=market.timestamp,symbol=market.symbol,signal=signal,raw_score=None,calibrated_probability=None,regime='UNAVAILABLE',entry_low=None,entry_high=None,stop_loss=None,tp1=None,tp2=None,tp3=None,risk_reward=None,grade='AVOID' if signal is Signal.AVOID else 'C',spread_state='UNKNOWN',portfolio_warning=None,pipeline_version=context.pipeline_version,operational_state=state)
