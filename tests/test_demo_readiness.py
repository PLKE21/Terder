from matamaple_trader.demo import DemoAuditMetrics, DemoReadinessGate, DemoReadinessPolicy


def healthy_metrics(**overrides):
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
    return DemoAuditMetrics(**values)


def test_demo_readiness_passes_healthy_evidence():
    result = DemoReadinessGate().evaluate(healthy_metrics())
    assert result.ready
    assert result.reasons == ()


def test_demo_readiness_requires_enough_evidence():
    result = DemoReadinessGate().evaluate(healthy_metrics(runtime_cycles=10, actionable_candidates=2, demo_orders_submitted=0))
    assert not result.ready
    assert "insufficient_runtime_cycles" in result.reasons
    assert "insufficient_actionable_candidates" in result.reasons


def test_demo_readiness_blocks_integrity_and_loss_failures():
    result = DemoReadinessGate().evaluate(
        healthy_metrics(journal_parse_errors=1, max_observed_drawdown_pct=11.0, max_daily_loss_pct=3.5)
    )
    assert not result.ready
    assert "journal_integrity_errors" in result.reasons
    assert "demo_drawdown_limit_exceeded" in result.reasons
    assert "demo_daily_loss_limit_exceeded" in result.reasons


def test_demo_readiness_blocks_order_failure_rate():
    policy = DemoReadinessPolicy(max_order_failure_rate=0.05)
    metrics = healthy_metrics(demo_orders_submitted=18, order_send_failures=1, order_check_rejections=1)
    result = DemoReadinessGate(policy).evaluate(metrics)
    assert not result.ready
    assert "order_failure_rate_exceeded" in result.reasons


def test_demo_readiness_blocks_ai_reliability_failures():
    metrics = healthy_metrics(ai_failures=3, ai_latency_violations=2)
    result = DemoReadinessGate().evaluate(metrics)
    assert not result.ready
    assert "ai_failure_rate_exceeded" in result.reasons
    assert "ai_latency_violation_rate_exceeded" in result.reasons
