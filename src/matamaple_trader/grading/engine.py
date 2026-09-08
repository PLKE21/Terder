from __future__ import annotations

from dataclasses import dataclass

from matamaple_trader.domain import Signal


@dataclass(frozen=True, slots=True)
class GradeConfig:
    version: str = "grade-v1"
    a_plus: float = 90.0
    a: float = 80.0
    a_minus: float = 72.0
    b: float = 62.0


@dataclass(frozen=True, slots=True)
class GradeInputs:
    signal: Signal
    calibrated_probability: float | None
    regime_ok: bool
    model_agreement: float
    trend_alignment: bool
    spread_ok: bool
    volatility_ok: bool
    structure_score: float
    risk_reward: float | None


@dataclass(frozen=True, slots=True)
class GradeResult:
    grade: str
    score: float
    version: str
    components: dict[str, float]


def grade_signal(inputs: GradeInputs, config: GradeConfig = GradeConfig()) -> GradeResult:
    if inputs.signal is Signal.AVOID:
        return GradeResult("AVOID", 0.0, config.version, {})
    if inputs.signal is Signal.WAIT:
        return GradeResult("C", 0.0, config.version, {})

    probability = 0.0 if inputs.calibrated_probability is None else min(max(inputs.calibrated_probability, 0.0), 1.0)
    agreement = min(max(inputs.model_agreement, 0.0), 1.0)
    structure = min(max(inputs.structure_score, 0.0), 1.0)
    rr = 0.0 if inputs.risk_reward is None else min(max(inputs.risk_reward / 3.0, 0.0), 1.0)

    components = {
        "calibrated_probability": probability * 30.0,
        "regime": 15.0 if inputs.regime_ok else 0.0,
        "model_agreement": agreement * 15.0,
        "trend_alignment": 10.0 if inputs.trend_alignment else 0.0,
        "spread": 10.0 if inputs.spread_ok else 0.0,
        "volatility": 5.0 if inputs.volatility_ok else 0.0,
        "structure": structure * 10.0,
        "risk_reward": rr * 5.0,
    }
    score = round(sum(components.values()), 6)
    if score >= config.a_plus:
        grade = "A+"
    elif score >= config.a:
        grade = "A"
    elif score >= config.a_minus:
        grade = "A-"
    elif score >= config.b:
        grade = "B"
    else:
        grade = "C"
    return GradeResult(grade, score, config.version, components)
