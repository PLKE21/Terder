from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from .dataset_review import review_collection_manifest


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description='Review collected XM dataset manifests before proposing or freezing a research holdout.')
    parser.add_argument('--manifest',default='data/historical/collection_manifest.json')
    args=parser.parse_args(argv)
    report=review_collection_manifest(args.manifest)
    payload=asdict(report)
    for key in ('common_start','common_end'):
        value=payload[key]
        payload[key]=value.isoformat() if value is not None else None
    for item in payload['items']:
        for key in ('first_available_timestamp','last_available_timestamp'):
            value=item[key]
            item[key]=value.isoformat() if value is not None else None
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0 if report.ready_for_holdout_proposal else 2


if __name__=='__main__':
    sys.exit(main())
