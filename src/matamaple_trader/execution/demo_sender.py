from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Protocol

from matamaple_trader.domain import Signal
from matamaple_trader.execution.order_adapter import ExecutionMode, OrderPlan, PreflightResult


@dataclass(frozen=True, slots=True)
class KillSwitch:
    engaged: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DemoSendResult:
    submitted: bool
    reason: str
    retcode: int | None = None
    order: int | None = None
    deal: int | None = None


class MT5TradeModule(Protocol):
    ACCOUNT_TRADE_MODE_DEMO: int
    TRADE_ACTION_DEAL: int
    ORDER_TYPE_BUY: int
    ORDER_TYPE_SELL: int
    ORDER_TIME_GTC: int

    def account_info(self) -> Any: ...
    def symbol_info(self, symbol: str) -> Any: ...
    def symbol_info_tick(self, symbol: str) -> Any: ...
    def order_check(self, request: dict[str, Any]) -> Any: ...
    def order_send(self, request: dict[str, Any]) -> Any: ...
    def last_error(self) -> Any: ...


class TradeJournal:
    def __init__(self, path: str | Path = "data/fbs/trades/demo_journal.jsonl") -> None:
        self.path = Path(path)

    def append(self, *, plan: OrderPlan, result: DemoSendResult, request: dict[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "plan": asdict(plan),
            "result": asdict(result),
            "request": request,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class FBSDemoOrderSender:
    """Guarded MT5 sender restricted to demo accounts.

    LIVE mode is rejected unconditionally. The caller must already have passed the
    deterministic Risk Engine, FBS Execution Guard, and order preflight.
    """

    def __init__(self, mt5: MT5TradeModule, *, journal: TradeJournal | None = None) -> None:
        self.mt5 = mt5
        self.journal = journal or TradeJournal()

    def _finish(self, plan: OrderPlan, result: DemoSendResult, request: dict[str, Any] | None = None) -> DemoSendResult:
        self.journal.append(plan=plan, result=result, request=request)
        return result

    def send(
        self,
        plan: OrderPlan,
        preflight: PreflightResult,
        *,
        mode: ExecutionMode,
        kill_switch: KillSwitch = KillSwitch(),
        comment: str = "MATAMAPLE_DEMO",
        magic: int = 560001,
    ) -> DemoSendResult:
        if mode is ExecutionMode.LIVE:
            return self._finish(plan, DemoSendResult(False, "live_trading_hard_disabled"))
        if mode is not ExecutionMode.DEMO:
            return self._finish(plan, DemoSendResult(False, "demo_sender_requires_demo_mode"))
        if kill_switch.engaged:
            return self._finish(plan, DemoSendResult(False, f"kill_switch:{kill_switch.reason or 'engaged'}"))
        if not preflight.allowed:
            return self._finish(plan, DemoSendResult(False, f"preflight:{preflight.reason}"))

        account = self.mt5.account_info()
        if account is None:
            return self._finish(plan, DemoSendResult(False, "account_info_unavailable"))
        if int(getattr(account, "trade_mode", -1)) != int(self.mt5.ACCOUNT_TRADE_MODE_DEMO):
            return self._finish(plan, DemoSendResult(False, "account_is_not_demo"))
        if not bool(getattr(account, "trade_allowed", True)):
            return self._finish(plan, DemoSendResult(False, "account_trade_disabled"))

        info = self.mt5.symbol_info(plan.symbol)
        tick = self.mt5.symbol_info_tick(plan.symbol)
        if info is None or tick is None:
            return self._finish(plan, DemoSendResult(False, "symbol_or_tick_unavailable"))

        is_buy = plan.side is Signal.BUY
        if plan.side not in {Signal.BUY, Signal.SELL}:
            return self._finish(plan, DemoSendResult(False, "invalid_side"))
        price = float(tick.ask if is_buy else tick.bid)
        deviation = int(plan.deviation_points)
        if deviation < 0:
            return self._finish(plan, DemoSendResult(False, "invalid_deviation"))

        request: dict[str, Any] = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": plan.symbol,
            "volume": float(plan.volume),
            "type": self.mt5.ORDER_TYPE_BUY if is_buy else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(plan.stop_loss),
            "deviation": deviation,
            "magic": int(magic),
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
        }
        if plan.take_profit is not None:
            request["tp"] = float(plan.take_profit)

        filling_mode = getattr(info, "filling_mode", None)
        if filling_mode is not None:
            request["type_filling"] = int(filling_mode)

        check = self.mt5.order_check(request)
        if check is None:
            return self._finish(plan, DemoSendResult(False, f"order_check_failed:{self.mt5.last_error()!r}"), request)
        check_retcode = int(getattr(check, "retcode", -1))
        check_comment = str(getattr(check, "comment", ""))
        if check_retcode != 0:
            return self._finish(plan, DemoSendResult(False, f"order_check_rejected:{check_retcode}:{check_comment}"), request)

        sent = self.mt5.order_send(request)
        if sent is None:
            return self._finish(plan, DemoSendResult(False, f"order_send_failed:{self.mt5.last_error()!r}"), request)
        retcode = int(getattr(sent, "retcode", -1))
        order = getattr(sent, "order", None)
        deal = getattr(sent, "deal", None)
        success_codes = {
            int(code)
            for code in (
                getattr(self.mt5, "TRADE_RETCODE_DONE", None),
                getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", None),
                getattr(self.mt5, "TRADE_RETCODE_PLACED", None),
            )
            if code is not None
        }
        if not success_codes or retcode not in success_codes:
            return self._finish(plan, DemoSendResult(False, f"order_send_rejected:{retcode}", retcode, order, deal), request)
        return self._finish(plan, DemoSendResult(True, "demo_order_submitted", retcode, order, deal), request)
