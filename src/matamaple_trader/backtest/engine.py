from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol

from matamaple_trader.domain import MarketSnapshot, PipelineContext, Signal, SignalResult


class SignalPipeline(Protocol):
    def evaluate(self, market: MarketSnapshot, context: PipelineContext) -> SignalResult: ...


@dataclass(frozen=True, slots=True)
class BacktestRecord:
    signal_timestamp: datetime
    execution_timestamp: datetime
    symbol: str
    signal: Signal
    entry_low: float | None
    entry_high: float | None
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    tp3: float | None
    grade: str
    pipeline_version: str


@dataclass(frozen=True, slots=True)
class BacktestReport:
    records: tuple[BacktestRecord, ...]

    @property
    def actionable_count(self) -> int:
        return sum(r.signal in {Signal.BUY, Signal.SELL} for r in self.records)


class BacktestEngine:
    """Historical replay only. All strategy decisions come from pipeline.evaluate()."""

    def __init__(self, pipeline: SignalPipeline) -> None:
        self.pipeline = pipeline

    def run(self, snapshots: Iterable[MarketSnapshot], context: PipelineContext) -> BacktestReport:
        records: list[BacktestRecord] = []
        last_timestamp: datetime | None = None
        for market in snapshots:
            if last_timestamp is not None and market.timestamp <= last_timestamp:
                raise ValueError("backtest snapshots must be strictly chronological")
            last_timestamp = market.timestamp
            result = self.pipeline.evaluate(market, context)
            records.append(
                BacktestRecord(
                    signal_timestamp=result.timestamp,
                    execution_timestamp=market.timestamp,
                    symbol=result.symbol,
                    signal=result.signal,
                    entry_low=result.entry_low,
                    entry_high=result.entry_high,
                    stop_loss=result.stop_loss,
                    tp1=result.tp1,
                    tp2=result.tp2,
                    tp3=result.tp3,
                    grade=result.grade,
                    pipeline_version=result.pipeline_version,
                )
            )
        return BacktestReport(tuple(records))
