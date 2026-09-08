from __future__ import annotations

from typing import Protocol

from matamaple_trader.domain import MarketSnapshot, OperationalState, PipelineContext, Signal, SignalResult
from matamaple_trader.grading import GradeInputs, grade_signal
from matamaple_trader.levels import build_price_levels
from matamaple_trader.portfolio import SignalExposure, exposure_warning
from matamaple_trader.signals import SignalGateConfig, SignalInputs, evaluate_signal


class PipelineStage(Protocol):
    def evaluate(self, market: MarketSnapshot, context: PipelineContext) -> SignalResult: ...


class SharedPipeline:
    """Safe skeleton used until all quantitative stages are configured."""

    def evaluate(self, market: MarketSnapshot, context: PipelineContext) -> SignalResult:
        state = context.operational_state
        signal = Signal.AVOID if state is OperationalState.PAUSED else Signal.WAIT
        return SignalResult(
            timestamp=market.timestamp,
            symbol=market.symbol,
            signal=signal,
            raw_score=None,
            calibrated_probability=None,
            regime="UNAVAILABLE",
            entry_low=None,
            entry_high=None,
            stop_loss=None,
            tp1=None,
            tp2=None,
            tp3=None,
            risk_reward=None,
            grade="AVOID" if signal is Signal.AVOID else "C",
            spread_state="UNKNOWN",
            portfolio_warning=None,
            pipeline_version=context.pipeline_version,
            operational_state=state,
        )


class QuantSignalPipeline:
    """Deterministic signal-only pipeline composition.

    Normalized model/regime/cost outputs are supplied through MarketSnapshot.metadata.
    This class has no execution or trade-placement API.
    """

    def __init__(self, gate_config: SignalGateConfig = SignalGateConfig()) -> None:
        self.gate_config = gate_config

    @staticmethod
    def _candidate(value: object) -> Signal:
        if isinstance(value, Signal):
            return value
        try:
            return Signal(str(value))
        except ValueError:
            return Signal.WAIT

    def evaluate(self, market: MarketSnapshot, context: PipelineContext) -> SignalResult:
        metadata = market.metadata
        candidate = self._candidate(metadata.get("candidate_signal", Signal.WAIT))
        raw_score = metadata.get("raw_score")
        probability = metadata.get("calibrated_probability")
        regime = str(metadata.get("regime", "UNAVAILABLE"))
        atr = metadata.get("atr")

        levels = None
        if candidate in {Signal.BUY, Signal.SELL} and isinstance(atr, (int, float)) and float(atr) > 0 and market.bars:
            levels = build_price_levels(candidate, market.bars, float(atr))

        decision = evaluate_signal(
            SignalInputs(
                candidate=candidate,
                calibrated_probability=float(probability) if isinstance(probability, (int, float)) else None,
                regime=regime,
                spread_points=float(metadata["spread_points"]) if isinstance(metadata.get("spread_points"), (int, float)) else None,
                normal_spread_points=float(metadata["normal_spread_points"]) if isinstance(metadata.get("normal_spread_points"), (int, float)) else None,
                expected_move=float(metadata["expected_move"]) if isinstance(metadata.get("expected_move"), (int, float)) else None,
                estimated_cost=float(metadata["estimated_cost"]) if isinstance(metadata.get("estimated_cost"), (int, float)) else None,
                risk_reward=levels.risk_reward if levels else None,
                mtf_aligned=bool(metadata.get("mtf_aligned", False)),
                operational_state=context.operational_state,
            ),
            self.gate_config,
        )

        actionable_levels = levels if decision.signal in {Signal.BUY, Signal.SELL} else None
        grade = grade_signal(
            GradeInputs(
                signal=decision.signal,
                calibrated_probability=float(probability) if isinstance(probability, (int, float)) else None,
                regime_ok=regime in self.gate_config.allowed_regimes,
                model_agreement=float(metadata.get("model_agreement", 0.0)),
                trend_alignment=bool(metadata.get("trend_alignment", False)),
                spread_ok=decision.spread_state == "NORMAL",
                volatility_ok=bool(metadata.get("volatility_ok", False)),
                structure_score=float(metadata.get("structure_score", 0.0)),
                risk_reward=actionable_levels.risk_reward if actionable_levels else None,
            )
        )

        exposures = metadata.get("portfolio_exposures")
        warning = exposure_warning(list(exposures)) if isinstance(exposures, (list, tuple)) and all(isinstance(item, SignalExposure) for item in exposures) else None

        return SignalResult(
            timestamp=market.timestamp,
            symbol=market.symbol,
            signal=decision.signal,
            raw_score=float(raw_score) if isinstance(raw_score, (int, float)) else None,
            calibrated_probability=float(probability) if isinstance(probability, (int, float)) else None,
            regime=regime,
            entry_low=actionable_levels.entry_low if actionable_levels else None,
            entry_high=actionable_levels.entry_high if actionable_levels else None,
            stop_loss=actionable_levels.stop_loss if actionable_levels else None,
            tp1=actionable_levels.tp1 if actionable_levels else None,
            tp2=actionable_levels.tp2 if actionable_levels else None,
            tp3=actionable_levels.tp3 if actionable_levels else None,
            risk_reward=actionable_levels.risk_reward if actionable_levels else None,
            grade=grade.grade,
            spread_state=decision.spread_state,
            portfolio_warning=warning,
            pipeline_version=context.pipeline_version,
            operational_state=context.operational_state,
        )
