from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from .holdout_evaluation import evaluate_frozen_holdout_once


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Evaluate the locked candidate on the frozen research holdout exactly once.')
    p.add_argument('--parquet',required=True)
    p.add_argument('--holdout-registry',default='data/validation/research_holdout.json')
    p.add_argument('--candidate-lock',default='data/validation/candidate_lock.json')
    p.add_argument('--output',default='artifacts/frozen_holdout_evaluation.json')
    p.add_argument('--confirm-one-shot',action='store_true')
    args=p.parse_args(argv)
    if not args.confirm_one_shot:
        print(json.dumps({'evaluated':False,'reason':'explicit --confirm-one-shot required'},indent=2))
        return 2
    report=evaluate_frozen_holdout_once(args.parquet,holdout_registry_path=args.holdout_registry,candidate_lock_path=args.candidate_lock,output_path=args.output)
    payload=asdict(report); payload['evaluated']=True; payload['output']=args.output
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    sys.exit(main())
