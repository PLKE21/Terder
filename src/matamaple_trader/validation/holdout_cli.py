from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from .holdout_proposal import load_coverages, propose_holdout


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description='Propose a research holdout from collected dataset manifests. This command never freezes or mutates holdout policy.')
    parser.add_argument('--manifest',default='data/historical/collection_manifest.json')
    parser.add_argument('--holdout-days',type=int,default=180)
    parser.add_argument('--minimum-development-days',type=int,default=365)
    args=parser.parse_args(argv)

    coverages=load_coverages(args.manifest)
    proposal=propose_holdout(
        coverages,
        holdout_days=args.holdout_days,
        minimum_development_days=args.minimum_development_days,
    )
    payload=asdict(proposal)
    for key in ('common_start','common_end','proposed_holdout_start','proposed_holdout_end'):
        value=payload[key]
        payload[key]=value.isoformat() if value is not None else None
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0 if proposal.ready_to_freeze else 2


if __name__=='__main__':
    sys.exit(main())
