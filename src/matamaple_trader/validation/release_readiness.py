from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from matamaple_trader.shadow.store import ShadowStore


@dataclass(frozen=True, slots=True)
class GateResult:
    ok: bool
    reasons: tuple[str,...]


def shadow_readiness(db_path:str|Path, *, min_predictions:int=200, min_outcome_coverage:float=0.8)->GateResult:
    if min_predictions<=0 or not (0<=min_outcome_coverage<=1):
        raise ValueError('invalid shadow readiness thresholds')
    stats=ShadowStore(db_path).stats()
    reasons=[]
    if stats['predictions']<min_predictions:
        reasons.append(f"insufficient_shadow_predictions:{stats['predictions']}<{min_predictions}")
    coverage=(stats['outcomes']/stats['predictions']) if stats['predictions'] else 0.0
    if coverage<min_outcome_coverage:
        reasons.append(f"insufficient_shadow_outcome_coverage:{coverage:.4f}<{min_outcome_coverage:.4f}")
    return GateResult(not reasons,tuple(reasons))


def demo_readiness(summary_path:str|Path, *, min_manual_fills:int=30, max_mean_abs_slippage_points:float=5.0)->GateResult:
    if min_manual_fills<=0 or max_mean_abs_slippage_points<0:
        raise ValueError('invalid demo readiness thresholds')
    p=Path(summary_path)
    if not p.is_file(): return GateResult(False,('missing_demo_summary',))
    payload=json.loads(p.read_text(encoding='utf-8'))
    reasons=[]
    count=int(payload.get('count',0))
    mean_abs=float(payload.get('mean_abs_slippage_points',float('inf')))
    if count<min_manual_fills: reasons.append(f'insufficient_manual_demo_fills:{count}<{min_manual_fills}')
    if mean_abs>max_mean_abs_slippage_points: reasons.append(f'demo_slippage_too_high:{mean_abs:.4f}>{max_mean_abs_slippage_points:.4f}')
    return GateResult(not reasons,tuple(reasons))


def release_readiness(*,holdout_report_path:str|Path,shadow_db_path:str|Path,demo_summary_path:str|Path,min_shadow_predictions:int=200,min_shadow_outcome_coverage:float=0.8,min_manual_fills:int=30,max_mean_abs_slippage_points:float=5.0)->dict[str,object]:
    reasons=[]
    hp=Path(holdout_report_path)
    holdout_ok=hp.is_file()
    if not holdout_ok: reasons.append('missing_frozen_holdout_evaluation')
    shadow=shadow_readiness(shadow_db_path,min_predictions=min_shadow_predictions,min_outcome_coverage=min_shadow_outcome_coverage)
    demo=demo_readiness(demo_summary_path,min_manual_fills=min_manual_fills,max_mean_abs_slippage_points=max_mean_abs_slippage_points)
    reasons.extend(shadow.reasons); reasons.extend(demo.reasons)
    return {
        'code_path_complete': True,
        'validated_for_real_use': not reasons,
        'holdout_evaluation_present': holdout_ok,
        'shadow_gate_ok': shadow.ok,
        'demo_gate_ok': demo.ok,
        'auto_trading_enabled': False,
        'reasons': reasons,
    }
