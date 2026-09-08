from matamaple_trader.domain import Signal
from matamaple_trader.execution import ExecutionDecision, ExecutionMode, FBSOrderAdapter, OrderPlan, size_volume_from_stop
from matamaple_trader.risk import RiskDecision


class FakePreflight:
    def __init__(self, margin=25.0, pnl=-10.0):
        self.margin = margin
        self.pnl = pnl

    def order_calc_margin(self, side, symbol, volume, price):
        return self.margin

    def order_calc_profit(self, side, symbol, volume, open_price, close_price):
        return self.pnl


def test_volume_sizing_respects_risk_and_step():
    lots = size_volume_from_stop(
        max_risk_amount=10.0,
        entry_price=1.1000,
        stop_loss=1.0990,
        tick_size=0.00001,
        tick_value=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert lots == 0.1


def test_preflight_accepts_within_risk_and_margin_but_dry_run_never_submits():
    adapter = FBSOrderAdapter(FakePreflight(margin=25.0, pnl=-9.9))
    plan = OrderPlan("EURUSD", Signal.BUY, 0.1, 1.1, 1.099, 1.102)
    risk = RiskDecision(True, "ok", 10.0)
    result = adapter.preflight_order(plan, risk, ExecutionDecision(True, "ok"), free_margin=100.0)
    assert result.allowed
    receipt = adapter.execute(plan, result)
    assert receipt.mode is ExecutionMode.DRY_RUN
    assert not receipt.submitted
    assert receipt.reason == "dry_run_validated_no_order_sent"


def test_preflight_blocks_risk_budget_exceeded():
    adapter = FBSOrderAdapter(FakePreflight(margin=25.0, pnl=-12.0))
    plan = OrderPlan("EURUSD", Signal.SELL, 0.1, 1.1, 1.101, 1.098)
    result = adapter.preflight_order(plan, RiskDecision(True, "ok", 10.0), ExecutionDecision(True, "ok"), free_margin=100.0)
    assert not result.allowed
    assert result.reason == "risk_budget_exceeded"


def test_preflight_blocks_insufficient_margin():
    adapter = FBSOrderAdapter(FakePreflight(margin=125.0, pnl=-9.0))
    plan = OrderPlan("EURUSD", Signal.BUY, 0.1, 1.1, 1.099, None)
    result = adapter.preflight_order(plan, RiskDecision(True, "ok", 10.0), ExecutionDecision(True, "ok"), free_margin=100.0)
    assert not result.allowed
    assert result.reason == "insufficient_margin"


def test_demo_and_live_are_hard_disabled():
    plan = OrderPlan("EURUSD", Signal.BUY, 0.1, 1.1, 1.099, None)
    for mode in (ExecutionMode.DEMO, ExecutionMode.LIVE):
        adapter = FBSOrderAdapter(FakePreflight(), mode=mode)
        receipt = adapter.execute(plan, adapter.preflight_order(plan, RiskDecision(True, "ok", 10.0), ExecutionDecision(True, "ok"), free_margin=100.0))
        assert not receipt.submitted
        assert receipt.reason == "order_send_disabled_until_demo_audit"
