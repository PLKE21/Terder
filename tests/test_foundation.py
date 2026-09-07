from __future__ import annotations

from pathlib import Path

import pytest

from matamaple_trader.config import RuntimeConfig
from matamaple_trader.domain import OperationalState, Signal


def test_required_signal_outcomes_exist() -> None:
    assert set(Signal) == {Signal.BUY, Signal.WAIT, Signal.SELL, Signal.AVOID}


def test_exact_operational_states_exist() -> None:
    assert set(OperationalState) == {
        OperationalState.ACTIVE,
        OperationalState.DEGRADED,
        OperationalState.PAUSED,
    }


def test_internal_timezone_is_utc_only() -> None:
    assert RuntimeConfig().operational_timezone == "UTC"
    with pytest.raises(ValueError):
        RuntimeConfig(operational_timezone="Asia/Bangkok")


def test_auto_trading_call_is_absent_from_source() -> None:
    source_root = Path(__file__).parents[1] / "src"
    forbidden = ".".join(("order", "send")) + "("
    for path in source_root.rglob("*.py"):
        assert forbidden not in path.read_text(encoding="utf-8"), f"Forbidden auto-trading call found in {path}"
