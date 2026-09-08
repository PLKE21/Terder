from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

from .candidate_lock import CandidateLock, CandidateLockRegistry, sha256_file


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Lock the development-selected candidate before any frozen holdout evaluation.')
    p.add_argument('--development-report',required=True)
    p.add_argument('--output',default='data/validation/candidate_lock.json')
    p.add_argument('--feature-version',default='features-v1')
    p.add_argument('--pipeline-version',default='v0.1.0')
    p.add_argument('--horizon-bars',type=int,default=4)
    p.add_argument('--neutral-threshold',type=float,default=0.0)
    p.add_argument('--purged-splits',type=int,default=5)
    p.add_argument('--embargo-bars',type=int,default=0)
    p.add_argument('--booster-n-estimators',type=int,default=100)
    p.add_argument('--confirm-lock',action='store_true')
    args=p.parse_args(argv)
    if not args.confirm_lock:
        print(json.dumps({'locked':False,'reason':'explicit --confirm-lock required'},indent=2))
        return 2
    report_path=Path(args.development_report)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('source_partition')!='development_only':
        raise RuntimeError('candidate lock requires a development-only experiment report')
    selected=str(report['selected_model'])
    lock=CandidateLock(
        candidate_id=f"{selected}:{report['label_version']}:{args.pipeline_version}",
        model_name=selected,
        label_name='n_bar_direction',
        label_version=str(report['label_version']),
        horizon_bars=args.horizon_bars,
        neutral_threshold=args.neutral_threshold,
        feature_version=args.feature_version,
        calibration_method='PLATT',
        purged_splits=args.purged_splits,
        embargo_bars=args.embargo_bars,
        booster_n_estimators=args.booster_n_estimators,
        development_report_sha256=sha256_file(report_path),
        pipeline_version=args.pipeline_version,
        locked_at=datetime.now(UTC),
    )
    CandidateLockRegistry(args.output).freeze(lock)
    print(json.dumps({'locked':True,'candidate_id':lock.candidate_id,'path':args.output},indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    sys.exit(main())
