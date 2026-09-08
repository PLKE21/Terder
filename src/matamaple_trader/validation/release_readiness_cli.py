from __future__ import annotations

import argparse
import json
import sys

from .release_readiness import release_readiness


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Summarize whether MATAMAPLE TRADER has enough real validation evidence for manual-use release candidate status.')
    p.add_argument('--holdout-report',default='artifacts/frozen_holdout_evaluation.json')
    p.add_argument('--shadow-db',default='data/shadow/shadow.sqlite')
    p.add_argument('--demo-summary',default='artifacts/xm_demo_summary.json')
    p.add_argument('--min-shadow-predictions',type=int,default=200)
    p.add_argument('--min-shadow-outcome-coverage',type=float,default=0.8)
    p.add_argument('--min-manual-fills',type=int,default=30)
    p.add_argument('--max-mean-abs-slippage-points',type=float,default=5.0)
    args=p.parse_args(argv)
    report=release_readiness(
        holdout_report_path=args.holdout_report,
        shadow_db_path=args.shadow_db,
        demo_summary_path=args.demo_summary,
        min_shadow_predictions=args.min_shadow_predictions,
        min_shadow_outcome_coverage=args.min_shadow_outcome_coverage,
        min_manual_fills=args.min_manual_fills,
        max_mean_abs_slippage_points=args.max_mean_abs_slippage_points,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report['validated_for_real_use'] else 2


if __name__=='__main__':
    sys.exit(main())
