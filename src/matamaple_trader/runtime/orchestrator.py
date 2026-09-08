from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
from time import monotonic
from typing import Protocol

from matamaple_trader.ai import (
    AICircuitBreaker,
    AIReview,
    AIReviewDecision,
    AIReviewInput,
    OllamaError,
    review_signal,
)
from matamaple_trader.domain import Signal, SignalResult
from matamaple_trader.execution import (
    ExecutionContext,
    ExecutionDecision,
    ExecutionMode,
    FBSDemoOrderSender,
    FBSExecutionGuard,
    FBSOrderAdapter,
    KillSwitch,
    OrderPlan,
    PreflightResult,
)
from matamaple_trader.risk import AccountRiskState, RiskDecision, RiskEngine


class Analyst(Protocol):
    def review(self, inputs: AIReviewInput) -> AIReview: ...


class RuntimeStage(StrEnum):
    QUANT = "QUANT"
    AI = "AI"
    RISK = "RISK"
    EXECUTION_GUARD = "EXECUTION_GUARD"
    PREFLIGHT = "PREFLIGHT"
    DEMO_SEND = "DEMO_SEND"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class RuntimeDecision:
    final_signal: Signal
    stage: RuntimeStage
    allowed: bool
    reason: str
    ai_review: AIReview | None = None
    ai_latency_seconds: float | None = None
    risk: RiskDecision | None = None
    execution: ExecutionDecision | None = None
    preflight: PreflightResult | None = None
    submitted: bool = False


class RuntimeAuditJournal:
    def __init__(self, path: str | Path = "data/fbs/runtime/demo_runtime.jsonl") -> None:
        self.path = Path(path)

    def append(self, *, signal: SignalResult, decision: RuntimeDecision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "quant": asdict(signal),
            "decision": asdict(decision),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class DemoRuntimeOrchestrator:
    """Single fail-closed runtime path for reviewed FBS demo execution."""

    def __init__(
        self,
        *,
        analyst: Analyst,
        risk_engine: RiskEngine,
        execution_guard: FBSExecutionGuard,
        order_adapter: FBSOrderAdapter,
        demo_sender: FBSDemoOrderSender,
        journal: RuntimeAuditJournal | None = None,
        max_ai_latency_seconds: float = 20.0,
        ai_circuit_breaker: AICircuitBreaker | None = None,
    ) -> None:
        if max_ai_latency_seconds <= 0:
            raise ValueError("max_ai_latency_seconds must be positive")
        self.analyst = analyst
        self.risk_engine = risk_engine
        self.execution_guard = execution_guard
        self.order_adapter = order_adapter
        self.demo_sender = demo_sender
        self.journal = journal or RuntimeAuditJournal()
        self.max_ai_latency_seconds = max_ai_latency_seconds
        self.ai_circuit_breaker = ai_circuit_breaker

    def _finish(self, signal: SignalResult, decision: RuntimeDecision) -> RuntimeDecision:
        self.journal.append(signal=signal, decision=decision)
        return decision

    def run(
        self,
        *,
        signal: SignalResult,
        ai_inputs: AIReviewInput,
        risk_state: AccountRiskState,
        execution_context: ExecutionContext,
        order_plan: OrderPlan,
        free_margin: float,
        mode: ExecutionMode = ExecutionMode.DRY_RUN,
        kill_switch: KillSwitch = KillSwitch(),
    ) -> RuntimeDecision:
        if signal.signal not in {Signal.BUY, Signal.SELL}:
            return self._finish(signal, RuntimeDecision(signal.signal, RuntimeStage.BLOCKED, False, "quant_signal_not_actionable"))
        if ai_inputs.quant_signal is not signal.signal or ai_inputs.symbol != signal.symbol:
            return self._finish(signal, RuntimeDecision(Signal.WAIT, RuntimeStage.BLOCKED, False, "ai_input_mismatch"))
        if order_plan.side is not signal.signal or order_plan.symbol != signal.symbol:
            return self._finish(signal, RuntimeDecision(Signal.WAIT, RuntimeStage.BLOCKED, False, "order_plan_mismatch"))

        if self.ai_circuit_breaker is not None and not self.ai_circuit_breaker.allow_request():
            state = self.ai_circuit_breaker.state()
            return self._finish(
                signal,
                RuntimeDecision(
                    Signal.WAIT,
                    RuntimeStage.AI,
                    False,
                    f"ai_circuit_open:{state.retry_after_seconds:.3f}",
                ),
            )

        started = monotonic()
        try:
            ai_review = self.analyst.review(ai_inputs)
        except (OllamaError, TimeoutError, RuntimeError) as exc:
            latency = monotonic() - started
            if self.ai_circuit_breaker is not None:
                self.ai_circuit_breaker.record_failure()
            return self._finish(
                signal,
                RuntimeDecision(Signal.WAIT, RuntimeStage.AI, False, f"ai_unavailable:{type(exc).__name__}", ai_latency_seconds=latency),
            )

        latency = monotonic() - started
        if latency > self.max_ai_latency_seconds:
            if self.ai_circuit_breaker is not None:
                self.ai_circuit_breaker.record_failure()
            return self._finish(
                signal,
                RuntimeDecision(Signal.WAIT, RuntimeStage.AI, False, "ai_latency_limit", ai_review, latency),
            )
        if self.ai_circuit_breaker is not None:
            self.ai_circuit_breaker.record_success()

        gated_signal = review_signal(signal, ai_review)
        if gated_signal not in {Signal.BUY, Signal.SELL}:
            reason = "ai_rejected" if ai_review.decision is AIReviewDecision.REJECT else "ai_not_confirmed"
            return self._finish(signal, RuntimeDecision(gated_signal, RuntimeStage.AI, False, reason, ai_review, latency))

        risk = self.risk_engine.evaluate(gated_signal, risk_state)
        if not risk.allowed:
            return self._finish(signal, RuntimeDecision(Signal.WAIT, RuntimeStage.RISK, False, risk.reason, ai_review, latency, risk))

        execution = self.execution_guard.evaluate(signal, risk, execution_context)
        if not execution.allowed:
            return self._finish(signal, RuntimeDecision(Signal.WAIT, RuntimeStage.EXECUTION_GUARD, False, execution.reason, ai_review, latency, risk, execution))

        preflight = self.order_adapter.preflight_order(order_plan, risk, execution, free_margin=free_margin)
        if not preflight.allowed:
            return self._finish(signal, RuntimeDecision(Signal.WAIT, RuntimeStage.PREFLIGHT, False, preflight.reason, ai_review, latency, risk, execution, preflight))

        if mode is ExecutionMode.DRY_RUN:
            receipt = self.order_adapter.execute(order_plan, preflight)
            return self._finish(signal, RuntimeDecision(gated_signal, RuntimeStage.PREFLIGHT, True, receipt.reason, ai_review, latency, risk, execution, preflight, False))

        send_result = self.demo_sender.send(order_plan, preflight, mode=mode, kill_switch=kill_switch)
        return self._finish(
            signal,
            RuntimeDecision(
                gated_signal if send_result.submitted else Signal.WAIT,
                RuntimeStage.DEMO_SEND,
                send_result.submitted,
                send_result.reason,
                ai_review,
                latency,
                risk,
                execution,
                preflight,
                send_result.submitted,
            ),
        )
