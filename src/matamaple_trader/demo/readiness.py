from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DemoAuditMetrics:
    runtime_cycles: int
    actionable_candidates: int
    demo_orders_submitted: int
    order_send_failures: int
    order_check_rejections: int
    ai_failures: int
    ai_latency_violations: int
    max_observed_drawdown_pct: float
    max_daily_loss_pct: float
    journal_parse_errors: int = 0


@dataclass(frozen=True, slots=True)
class DemoReadinessPolicy:
    min_runtime_cycles: int = 100
    min_actionable_candidates: int = 20
    max_order_failure_rate: float = 0.02
    max_ai_failure_rate: float = 0.05
    max_ai_latency_violation_rate: float = 0.02
    max_demo_drawdown_pct: float = 10.0
    max_demo_daily_loss_pct: float = 3.0

    def __post_init__(self) -> None:
        if self.min_runtime_cycles < 1 or self.min_actionable_candidates < 1:
            raise ValueError("minimum audit sample sizes must be positive")
        for value in (
            self.max_order_failure_rate,
            self.max_ai_failure_rate,
            self.max_ai_latency_violation_rate,
        ):
            if not 0 <= value <= 1:
                raise ValueError("failure-rate thresholds must be between 0 and 1")
        if self.max_demo_drawdown_pct <= 0 or self.max_demo_daily_loss_pct <= 0:
            raise ValueError("loss thresholds must be positive")


@dataclass(frozen=True, slots=True)
class DemoReadinessResult:
    ready: bool
    reasons: tuple[str, ...]


class DemoReadinessGate:
    """Deterministic release gate from dry-run/demo evidence to extended demo testing.

    Passing this gate does not authorize live trading. It only indicates that the
    recorded FBS demo runtime evidence satisfies the configured engineering thresholds.
    """

    def __init__(self, policy: DemoReadinessPolicy = DemoReadinessPolicy()) -> None:
        self.policy = policy

    def evaluate(self, metrics: DemoAuditMetrics) -> DemoReadinessResult:
        reasons: list[str] = []
        p = self.policy

        if metrics.runtime_cycles < p.min_runtime_cycles:
            reasons.append("insufficient_runtime_cycles")
        if metrics.actionable_candidates < p.min_actionable_candidates:
            reasons.append("insufficient_actionable_candidates")
        if metrics.journal_parse_errors > 0:
            reasons.append("journal_integrity_errors")
        if metrics.max_observed_drawdown_pct > p.max_demo_drawdown_pct:
            reasons.append("demo_drawdown_limit_exceeded")
        if metrics.max_daily_loss_pct > p.max_demo_daily_loss_pct:
            reasons.append("demo_daily_loss_limit_exceeded")

        order_attempts = metrics.demo_orders_submitted + metrics.order_send_failures + metrics.order_check_rejections
        if order_attempts > 0:
            order_failure_rate = (metrics.order_send_failures + metrics.order_check_rejections) / order_attempts
            if order_failure_rate > p.max_order_failure_rate:
                reasons.append("order_failure_rate_exceeded")
        elif metrics.actionable_candidates >= p.min_actionable_candidates:
            reasons.append("no_demo_order_attempts")

        ai_denominator = max(metrics.actionable_candidates, 1)
        if metrics.ai_failures / ai_denominator > p.max_ai_failure_rate:
            reasons.append("ai_failure_rate_exceeded")
        if metrics.ai_latency_violations / ai_denominator > p.max_ai_latency_violation_rate:
            reasons.append("ai_latency_violation_rate_exceeded")

        return DemoReadinessResult(not reasons, tuple(reasons))
