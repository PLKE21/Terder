from __future__ import annotations

import argparse
import json
import sys

from matamaple_trader.data.development_builder import build_development_dataset
from matamaple_trader.experiments.development import run_development_experiment, write_experiment_report
from matamaple_trader.labels import n_bar_direction_contract
from matamaple_trader.models.calibration import CalibrationMethod


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(
        description='Run development-only Purged CV/CPCV/Walk-Forward model research. Frozen holdout outcomes are never loaded.'
    )
    parser.add_argument('--parquet',required=True)
    parser.add_argument('--holdout-registry',default='data/policies/research_holdout.json')
    parser.add_argument('--horizon-bars',type=int,default=4)
    parser.add_argument('--purged-splits',type=int,default=5)
    parser.add_argument('--embargo-bars',type=int,default=0)
    parser.add_argument('--cpcv-groups',type=int,default=6)
    parser.add_argument('--cpcv-test-groups',type=int,default=2)
    parser.add_argument('--wf-min-train-bars',type=int,default=2000)
    parser.add_argument('--wf-test-bars',type=int,default=250)
    parser.add_argument('--wf-step-bars',type=int,default=None)
    parser.add_argument('--calibration',choices=('PLATT','ISOTONIC'),default='PLATT')
    parser.add_argument('--booster-n-estimators',type=int,default=100)
    parser.add_argument('--output',default='artifacts/development_experiment.json')
    args=parser.parse_args(argv)

    contract=n_bar_direction_contract(horizon_bars=args.horizon_bars,neutral_threshold=0.0)
    dataset=build_development_dataset(
        args.parquet,
        holdout_registry_path=args.holdout_registry,
        label_contract=contract,
    )
    report=run_development_experiment(
        dataset,
        label_contract=contract,
        holdout_registry_path=args.holdout_registry,
        purged_splits=args.purged_splits,
        embargo_bars=args.embargo_bars,
        cpcv_groups=args.cpcv_groups,
        cpcv_test_groups=args.cpcv_test_groups,
        wf_min_train_bars=args.wf_min_train_bars,
        wf_test_bars=args.wf_test_bars,
        wf_step_bars=args.wf_step_bars,
        calibration_method=CalibrationMethod(args.calibration),
        booster_n_estimators=args.booster_n_estimators,
    )
    path=write_experiment_report(report,args.output)
    print(json.dumps({
        'source_partition':report.source_partition,
        'development_rows':report.development_rows,
        'selected_model':report.selected_model,
        'selection_rule':report.selection_rule,
        'report_path':str(path),
        'holdout_evaluated':False,
    },indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    sys.exit(main())
