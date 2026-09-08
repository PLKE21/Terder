from datetime import UTC, datetime

from matamaple_trader.ai import AIReview, AIReviewDecision, AIReviewInput, OllamaError
from matamaple_trader.domain import OperationalState, Signal, SignalResult
from matamaple_trader.execution import (
    DemoSendResult,
    ExecutionContext,
    ExecutionDecision,
    ExecutionMode,
    OrderPlan,
    PreflightResult,
)
from matamaple_trader.risk import AccountRiskState, RiskDecision
from matamaple_trader.runtime import DemoRuntimeOrchestrator, RuntimeStage


class MemoryJournal:
    def __init__(self):
        self.records = []

    def append(self, *, signal, decision):
        self.records.append((signal, decision))


class FakeAnalyst:
    def __init__(self, decision=AIReviewDecision.CONFIRM, error=None):
        self.decision = decision
        self.error = error
        self.calls = 0

    def review(self, inputs):
        self.calls += 1
        if self.error:
            raise self.error
        return AIReview(self.decision, 0.8, "test", "fake")


class FakeRisk:
    def __init__(self, allowed=True):
        self.allowed = allowed

    def evaluate(self, signal, state):
        return RiskDecision(self.allowed, "ok" if self.allowed else "daily_loss_limit", 10.0 if self.allowed else 0.0)


class FakeGuard:
    def __init__(self, allowed=True):
        self.allowed = allowed

    def evaluate(self, signal, risk, ctx):
        return ExecutionDecision(self.allowed, "ok" if self.allowed else "spread_limit")


class FakeAdapter:
    def __init__(self, allowed=True):
        self.allowed = allowed

    def preflight_order(self, plan, risk, execution, *, free_margin):
        return PreflightResult(self.allowed, "ok" if self.allowed else "insufficient_margin", 10.0, 5.0)

    def execute(self, plan, preflight):
        class Receipt:
            submitted = False
            reason = "dry_run_validated_no_order_sent"
        return Receipt()


class FakeSender:
    def __init__(self, submitted=True):
        self.submitted = submitted
        self.calls = 0

    def send(self, plan, preflight, *, mode, kill_switch):
        self.calls += 1
        return DemoSendResult(self.submitted, "demo_order_submitted" if self.submitted else "blocked")


def signal_result(signal=Signal.BUY):
    return SignalResult(
        timestamp=datetime.now(UTC),
        symbol="EURUSD",
        signal=signal,
        raw_score=0.8,
        calibrated_probability=0.75,
        regime="trend",
        entry_low=1.1000,
        entry_high=1.1002,
        stop_loss=1.0990,
        tp1=1.1020,
        tp2=None,
        tp3=None,
        risk_reward=2.0,
        grade="A",
        spread_state="OK",
        portfolio_warning=None,
        pipeline_version="test",
        operational_state=OperationalState.ACTIVE,
    )


def ai_input(signal=Signal.BUY):
    return AIReviewInput("EURUSD", signal, 0.75, "trend", "A", "OK", 2.0, True, True, 0.8)


def account_state():
    return AccountRiskState(1000.0, 1000.0, 0.0, 1000.0, 900.0, 0)


def exec_context():
    return ExecutionContext(10.0, 30.0, 100.0, 20.0, 0.0)


def plan(side=Signal.BUY):
    return OrderPlan("EURUSD", side, 0.1, 1.1001, 1.0990, 1.1020)


def build(analyst=None, risk=None, guard=None, adapter=None, sender=None):
    journal = MemoryJournal()
    runtime = DemoRuntimeOrchestrator(
        analyst=analyst or FakeAnalyst(),
        risk_engine=risk or FakeRisk(),
        execution_guard=guard or FakeGuard(),
        order_adapter=adapter or FakeAdapter(),
        demo_sender=sender or FakeSender(),
        journal=journal,
    )
    return runtime, journal


def run(runtime, *, signal=None, inputs=None, mode=ExecutionMode.DRY_RUN):
    s = signal or signal_result()
    return runtime.run(
        signal=s,
        ai_inputs=inputs or ai_input(s.signal),
        risk_state=account_state(),
        execution_context=exec_context(),
        order_plan=plan(s.signal if s.signal in {Signal.BUY, Signal.SELL} else Signal.BUY),
        free_margin=900.0,
        mode=mode,
    )


def test_wait_signal_never_calls_ai():
    analyst = FakeAnalyst()
    runtime, journal = build(analyst=analyst)
    result = run(runtime, signal=signal_result(Signal.WAIT), inputs=ai_input(Signal.WAIT))
    assert not result.allowed
    assert result.reason == "quant_signal_not_actionable"
    assert analyst.calls == 0
    assert len(journal.records) == 1


def test_ai_reject_stops_before_risk():
    runtime, _ = build(analyst=FakeAnalyst(AIReviewDecision.REJECT))
    result = run(runtime)
    assert not result.allowed
    assert result.stage is RuntimeStage.AI
    assert result.final_signal is Signal.WAIT


def test_ai_error_fails_closed():
    runtime, _ = build(analyst=FakeAnalyst(error=OllamaError("offline")))
    result = run(runtime)
    assert not result.allowed
    assert result.stage is RuntimeStage.AI
    assert result.reason.startswith("ai_unavailable")


def test_risk_block_stops_flow():
    runtime, _ = build(risk=FakeRisk(False))
    result = run(runtime)
    assert not result.allowed
    assert result.stage is RuntimeStage.RISK
    assert result.reason == "daily_loss_limit"


def test_execution_guard_block_stops_flow():
    runtime, _ = build(guard=FakeGuard(False))
    result = run(runtime)
    assert not result.allowed
    assert result.stage is RuntimeStage.EXECUTION_GUARD
    assert result.reason == "spread_limit"


def test_preflight_block_stops_flow():
    runtime, _ = build(adapter=FakeAdapter(False))
    result = run(runtime)
    assert not result.allowed
    assert result.stage is RuntimeStage.PREFLIGHT
    assert result.reason == "insufficient_margin"


def test_dry_run_validates_without_submission():
    sender = FakeSender(True)
    runtime, _ = build(sender=sender)
    result = run(runtime, mode=ExecutionMode.DRY_RUN)
    assert result.allowed
    assert not result.submitted
    assert result.reason == "dry_run_validated_no_order_sent"
    assert sender.calls == 0


def test_demo_submits_only_after_all_gates_pass():
    sender = FakeSender(True)
    runtime, _ = build(sender=sender)
    result = run(runtime, mode=ExecutionMode.DEMO)
    assert result.allowed
    assert result.submitted
    assert result.stage is RuntimeStage.DEMO_SEND
    assert sender.calls == 1


def test_signal_and_ai_input_must_match():
    runtime, _ = build()
    result = run(runtime, inputs=ai_input(Signal.SELL))
    assert not result.allowed
    assert result.reason == "ai_input_mismatch"
