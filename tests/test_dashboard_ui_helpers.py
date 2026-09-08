from pathlib import Path

from matamaple_trader.dashboard.app import _latest_runtime_summary, _load_runtime_records, _signal_class


def test_runtime_records_fail_safe_when_missing(tmp_path: Path) -> None:
    assert _load_runtime_records(tmp_path / "missing.jsonl") == []


def test_runtime_summary_defaults_to_wait() -> None:
    summary = _latest_runtime_summary([])
    assert summary["signal"] == "WAIT"
    assert summary["execution"] == "DRY RUN"


def test_runtime_summary_reads_latest_decision() -> None:
    records = [
        {
            "quant": {"symbol": "EURUSD", "signal": "BUY"},
            "decision": {
                "final_signal": "BUY",
                "reason": "dry_run_validated_no_order_sent",
                "submitted": False,
                "risk": {"allowed": True},
                "ai_review": {"decision": "CONFIRM"},
            },
        }
    ]
    summary = _latest_runtime_summary(records)
    assert summary["symbol"] == "EURUSD"
    assert summary["signal"] == "BUY"
    assert summary["ai"] == "CONFIRM"
    assert summary["risk"] == "PASS"
    assert summary["execution"] == "SAFE"


def test_signal_class_is_simple_and_predictable() -> None:
    assert _signal_class("BUY") == "mm-good"
    assert _signal_class("SELL") == "mm-good"
    assert _signal_class("WAIT") == "mm-warn"
    assert _signal_class("AVOID") == "mm-bad"
