from __future__ import annotations
from matamaple_trader.domain import MarketSnapshot, PipelineContext, SignalResult
from matamaple_trader.pipeline.core import SharedPipeline
class HistoricalAdapter:
    def __init__(self,pipeline:SharedPipeline): self.pipeline=pipeline
    def evaluate(self,market:MarketSnapshot,context:PipelineContext)->SignalResult: return self.pipeline.evaluate(market,context)
class MockLiveAdapter(HistoricalAdapter): pass
