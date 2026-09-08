from __future__ import annotations

import argparse
import json
from pathlib import Path

from .readiness import DemoAuditMetrics, DemoReadinessGate, DemoReadinessPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MATAMAPLE TRADER FBS demo audit readiness")
    parser.add_argument("--metrics", required=True, help="Path to JSON file containing DemoAuditMetrics fields")
    parser.add_argument("--min-runtime-cycles", type=int, default=100)
    parser.add_argument("--min-actionable-candidates", type=int, default=20)
    parser.add_argument("--max-order-failure-rate", type=float, default=0.02)
    parser.add_argument("--max-ai-failure-rate", type=float, default=0.05)
    parser.add_argument("--max-ai-latency-violation-rate", type=float, default=0.02)
    parser.add_argument("--max-demo-drawdown-pct", type=float, default=10.0)
    parser.add_argument("--max-demo-daily-loss-pct", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metrics = DemoAuditMetrics(**payload)
    policy = DemoReadinessPolicy(
        min_runtime_cycles=args.min_runtime_cycles,
        min_actionable_candidates=args.min_actionable_candidates,
        max_order_failure_rate=args.max_order_failure_rate,
        max_ai_failure_rate=args.max_ai_failure_rate,
        max_ai_latency_violation_rate=args.max_ai_latency_violation_rate,
        max_demo_drawdown_pct=args.max_demo_drawdown_pct,
        max_demo_daily_loss_pct=args.max_demo_daily_loss_pct,
    )
    result = DemoReadinessGate(policy).evaluate(metrics)
    print(json.dumps({"ready": result.ready, "reasons": list(result.reasons)}, indent=2))
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
