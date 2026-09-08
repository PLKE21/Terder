import json

from matamaple_trader.demo.readiness_cli import main


def _metrics(**overrides):
    values = dict(
        runtime_cycles=200,
        actionable_candidates=40,
        demo_orders_submitted=38,
        order_send_failures=0,
        order_check_rejections=0,
        ai_failures=0,
        ai_latency_violations=0,
        max_observed_drawdown_pct=2.0,
        max_daily_loss_pct=1.0,
        journal_parse_errors=0,
    )
    values.update(overrides)
    return values


def test_demo_readiness_cli_returns_zero_for_ready_metrics(tmp_path, capsys):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_metrics()), encoding="utf-8")
    assert main(["--metrics", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True


def test_demo_readiness_cli_returns_two_for_failed_metrics(tmp_path, capsys):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_metrics(runtime_cycles=2)), encoding="utf-8")
    assert main(["--metrics", str(path)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert "insufficient_runtime_cycles" in payload["reasons"]
