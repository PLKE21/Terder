from types import SimpleNamespace

from matamaple_trader.domain import Signal
from matamaple_trader.execution import (
    ExecutionMode,
    FBSDemoOrderSender,
    KillSwitch,
    OrderPlan,
    PreflightResult,
    TradeJournal,
)


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_PLACED = 10008

    def __init__(self, *, trade_mode=0, check_retcode=0, send_retcode=10009):
        self.trade_mode = trade_mode
        self.check_retcode = check_retcode
        self.send_retcode = send_retcode
        self.send_calls = 0

    def account_info(self):
        return SimpleNamespace(trade_mode=self.trade_mode, trade_allowed=True)

    def symbol_info(self, symbol):
        return SimpleNamespace(filling_mode=1)

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=1.0999, ask=1.1001)

    def order_check(self, request):
        return SimpleNamespace(retcode=self.check_retcode, comment="ok" if self.check_retcode == 0 else "rejected")

    def order_send(self, request):
        self.send_calls += 1
        return SimpleNamespace(retcode=self.send_retcode, order=123, deal=456)

    def last_error(self):
        return (0, "ok")


def _plan():
    return OrderPlan("EURUSD", Signal.BUY, 0.1, 1.1001, 1.0990, 1.1020, deviation_points=10)


def _preflight():
    return PreflightResult(True, "ok", 25.0, 9.0)


def test_demo_sender_submits_only_after_successful_order_check(tmp_path):
    mt5 = FakeMT5()
    sender = FBSDemoOrderSender(mt5, journal=TradeJournal(tmp_path / "journal.jsonl"))
    result = sender.send(_plan(), _preflight(), mode=ExecutionMode.DEMO)
    assert result.submitted
    assert result.reason == "demo_order_submitted"
    assert mt5.send_calls == 1
    assert (tmp_path / "journal.jsonl").exists()


def test_live_mode_is_hard_disabled(tmp_path):
    mt5 = FakeMT5()
    sender = FBSDemoOrderSender(mt5, journal=TradeJournal(tmp_path / "journal.jsonl"))
    result = sender.send(_plan(), _preflight(), mode=ExecutionMode.LIVE)
    assert not result.submitted
    assert result.reason == "live_trading_hard_disabled"
    assert mt5.send_calls == 0


def test_kill_switch_blocks_before_order_check_and_send(tmp_path):
    mt5 = FakeMT5()
    sender = FBSDemoOrderSender(mt5, journal=TradeJournal(tmp_path / "journal.jsonl"))
    result = sender.send(_plan(), _preflight(), mode=ExecutionMode.DEMO, kill_switch=KillSwitch(True, "daily_guard"))
    assert not result.submitted
    assert result.reason == "kill_switch:daily_guard"
    assert mt5.send_calls == 0


def test_real_account_is_rejected_even_in_demo_mode(tmp_path):
    mt5 = FakeMT5(trade_mode=2)
    sender = FBSDemoOrderSender(mt5, journal=TradeJournal(tmp_path / "journal.jsonl"))
    result = sender.send(_plan(), _preflight(), mode=ExecutionMode.DEMO)
    assert not result.submitted
    assert result.reason == "account_is_not_demo"
    assert mt5.send_calls == 0


def test_order_check_rejection_blocks_order_send(tmp_path):
    mt5 = FakeMT5(check_retcode=10016)
    sender = FBSDemoOrderSender(mt5, journal=TradeJournal(tmp_path / "journal.jsonl"))
    result = sender.send(_plan(), _preflight(), mode=ExecutionMode.DEMO)
    assert not result.submitted
    assert result.reason.startswith("order_check_rejected:10016")
    assert mt5.send_calls == 0


def test_non_success_retcode_is_reported(tmp_path):
    mt5 = FakeMT5(send_retcode=10021)
    sender = FBSDemoOrderSender(mt5, journal=TradeJournal(tmp_path / "journal.jsonl"))
    result = sender.send(_plan(), _preflight(), mode=ExecutionMode.DEMO)
    assert not result.submitted
    assert result.reason == "order_send_rejected:10021"
    assert mt5.send_calls == 1
