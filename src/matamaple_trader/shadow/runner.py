from __future__ import annotations

from matamaple_trader.domain import MarketSnapshot, PipelineContext, SignalResult
from .store import ShadowStore


class ShadowRunner:
    def __init__(self, pipeline, store: ShadowStore) -> None:
        self.pipeline = pipeline
        self.store = store

    def evaluate_and_store(self, market: MarketSnapshot, context: PipelineContext, *, observation: dict[str, object] | None = None) -> tuple[SignalResult, int]:
        result = self.pipeline.evaluate(market, context)
        prediction_id = self.store.append_prediction(result, observation=observation)
        return result, prediction_id
