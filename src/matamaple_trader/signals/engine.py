from __future__ import annotations

from dataclasses import dataclass

from matamaple_trader.domain import OperationalState, Signal


@dataclass(frozen=True, slots=True)
class SignalGateConfig:
    min_probability: float = 0.60
    max_spread_ratio: float = 1.50
    min_expected_move_cost_ratio: float = 1.50
    min_risk_reward: float = 1.50
    allowed_regimes: tuple[str, ...] = ("TREND", "NORMAL")
    require_mtf_alignment: bool = True


@dataclass(frozen=True, slots=True)
class SignalInputs:
    candidate: Signal
    calibrated_probability: float | None
    regime: str
    spread_points: float | None
    normal_spread_points: float | None
    expected_move: float | None
    estimated_cost: float | None
    risk_reward: float | None
    mtf_aligned: bool
    operational_state: OperationalState = OperationalState.ACTIVE


@dataclass(frozen=True, slots=True)
class SignalDecision:
    signal: Signal
    reasons: tuple[str, ...]
    spread_state: str


def evaluate_signal(inputs: SignalInputs, config: SignalGateConfig = SignalGateConfig()) -> SignalDecision:
    if inputs.operational_state is OperationalState.PAUSED:
        return SignalDecision(Signal.AVOID, ("operational_paused",), "UNKNOWN")
    if inputs.operational_state is OperationalState.DEGRADED:
        return SignalDecision(Signal.WAIT, ("operational_degraded",), "UNKNOWN")
    if inputs.candidate not in {Signal.BUY, Signal.SELL}:
        return SignalDecision(Signal.WAIT, ("no_actionable_candidate",), "UNKNOWN")

    reasons: list[str] = []
    spread_state = "UNKNOWN"
    if inputs.calibrated_probability is None or inputs.calibrated_probability < config.min_probability:
        reasons.append("probability_gate")
    if inputs.regime not in config.allowed_regimes:
        reasons.append("regime_gate")

    if inputs.spread_points is not None and inputs.normal_spread_points and inputs.normal_spread_points > 0:
        ratio = inputs.spread_points / inputs.normal_spread_points
        spread_state = "HIGH" if ratio > config.max_spread_ratio else "NORMAL"
        if spread_state == "HIGH":
            return SignalDecision(Signal.AVOID, ("spread_gate",), spread_state)
    else:
        reasons.append("spread_unknown")

    if inputs.expected_move is None or inputs.estimated_cost is None or inputs.estimated_cost < 0:
        reasons.append("cost_gate_unknown")
    elif inputs.expected_move <= inputs.estimated_cost * config.min_expected_move_cost_ratio:
        reasons.append("cost_gate")

    if inputs.risk_reward is None or inputs.risk_reward < config.min_risk_reward:
        reasons.append("risk_reward_gate")
    if config.require_mtf_alignment and not inputs.mtf_aligned:
        reasons.append("mtf_gate")

    if reasons:
        return SignalDecision(Signal.WAIT, tuple(reasons), spread_state)
    return SignalDecision(inputs.candidate, (), spread_state)
