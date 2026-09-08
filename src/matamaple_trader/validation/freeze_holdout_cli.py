from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import sys
from pathlib import Path

from .holdout import HoldoutPolicy, HoldoutRegistry
from .holdout_proposal import load_coverages, propose_holdout


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description='Explicitly freeze a research holdout after reviewing clean real-data manifests. This action is irreversible for the target registry path.')
    parser.add_argument('--manifest',default='data/historical/collection_manifest.json')
    parser.add_argument('--registry',default='data/validation/research_holdout.json')
    parser.add_argument('--holdout-days',type=int,default=180)
    parser.add_argument('--minimum-development-days',type=int,default=365)
    parser.add_argument('--policy-version',required=True)
    parser.add_argument('--confirm-freeze',action='store_true',help='Required explicit acknowledgement that the proposed boundary has been reviewed and must never be tuned/moved after freezing.')
    args=parser.parse_args(argv)

    if not args.confirm_freeze:
        parser.error('--confirm-freeze is required; run matamaple-holdout-proposal and review the boundary first')

    proposal=propose_holdout(
        load_coverages(args.manifest),
        holdout_days=args.holdout_days,
        minimum_development_days=args.minimum_development_days,
    )
    if not proposal.ready_to_freeze or proposal.proposed_holdout_start is None or proposal.proposed_holdout_end is None:
        print(json.dumps({'frozen':False,'reasons':proposal.reasons},indent=2,sort_keys=True))
        return 2

    registry=HoldoutRegistry(Path(args.registry))
    policy=HoldoutPolicy(
        holdout_start=proposal.proposed_holdout_start,
        holdout_end=proposal.proposed_holdout_end,
        freeze_timestamp=datetime.now(UTC),
        policy_version=args.policy_version,
        kind='research_holdout',
    )
    registry.freeze(policy)
    payload=registry.load()
    print(json.dumps({'frozen':True,'registry':str(args.registry),'policy':payload},indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    sys.exit(main())
