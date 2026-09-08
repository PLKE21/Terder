from datetime import UTC, datetime

from matamaple_trader.ai import AICircuitBreaker, AIReview, AIReviewDecision, AIReviewInput, OllamaError
from matamaple_trader.domain import OperationalState, Signal, SignalResult
from matamaple_trader.execution import DemoSendResult, ExecutionContext, ExecutionDecision, ExecutionMode, OrderPlan, PreflightResult
from matamaple_trader.risk import AccountRiskState, RiskDecision
from matamaple_trader.runtime import DemoRuntimeOrchestrator, RuntimeStage


class MemoryJournal:
    def append(self, *, signal, decision):
        pass


class FlakyAnalyst:
    def __init__(self):
        self.calls = 0
        self.fail = True

    def review(self, inputs):
        self.calls += 1
        if self.fail:
            raise OllamaError("offline")
        return AIReview(AIReviewDecision.CONFIRM, 0.9, "ok", "fake")


class FakeRisk:
    def evaluate(self, signal, state):
        return RiskDecision(True, "ok", 10.0)


class FakeGuard:
    def evaluate(self, signal, risk, ctx):
        return ExecutionDecision(True, "ok")


class FakeAdapter:
    def preflight_order(self, plan, risk, execution, *, free_margin):
        return PreflightResult(True, "ok", 10.0, 5.0)

    def execute(self, plan, preflight):
        class Receipt:
            submitted = False
            reason = "dry_run_validated_no_order_sent"
        return Receipt()


class FakeSender:
    def send(self, plan, preflight, *, mode, kill_switch):
        return DemoSendResult(True, "demo_order_submitted")


def _signal():
    return SignalResult(
        datetime.now(UTC), "EURUSD", Signal.BUY, 0.8, 0.75, "trend",
        1.1000, 1.1002, 1.0990, 1.1020, None, None, 2.0, "A", "OK", None,
        "test", OperationalState.ACTIVE,
    )


def _run(runtime):
    return runtime.run(
        signal=_signal(),
        ai_inputs=AIReviewInput("EURUSD", Signal.BUY, 0.75, "trend", "A", "OK", 2.0, True, True, 0.8),
        risk_state=AccountRiskState(1000.0, 1000.0, 0.0, 1000.0, 900.0, 0),
        execution_context=ExecutionContext(10.0, 30.0, 100.0, 20.0, 0.0),
        order_plan=OrderPlan("EURUSD", Signal.BUY, 0.1, 1.1001, 1.0990, 1.1020),
        free_margin=900.0,
        mode=ExecutionMode.DRY_RUN,
    )


def test_runtime_opens_ai_circuit_and_stops_new_calls():
    analyst = FlakyAnalyst()
    circuit = AICircuitBreaker(max_failures=2, cooldown_seconds=60.0)
    runtime = DemoRuntimeOrchestrator(
        analyst=analyst,
        risk_engine=FakeRisk(),
        execution_guard=FakeGuard(),
        order_adapter=FakeAdapter(),
        demo_sender=FakeSender(),
        journal=MemoryJournal(),
        ai_circuit_breaker=circuit,
    )

    first = _run(runtime)
    second = _run(runtime)
    third = _run(runtime)

    assert first.reason.startswith("ai_unavailable")
    assert second.reason.startswith("ai_unavailable")
    assert third.stage is RuntimeStage.AI
    assert third.reason.startswith("ai_circuit_open")
    assert analyst.calls == 2
