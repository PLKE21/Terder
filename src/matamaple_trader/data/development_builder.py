from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from matamaple_trader.features import FeatureEngine
from matamaple_trader.labels import LabelContract, make_labels
from matamaple_trader.validation.holdout import HoldoutRegistry
from matamaple_trader.validation.leakage import assert_no_label_features, assert_monotonic_time


@dataclass(frozen=True, slots=True)
class DevelopmentDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    label_column: str
    cutoff_timestamp: datetime


def build_development_dataset(
    parquet_path: str | Path,
    *,
    holdout_registry_path: str | Path,
    label_contract: LabelContract,
    feature_engine: FeatureEngine | None = None,
) -> DevelopmentDataset:
    """Build development-only features/labels without reading holdout rows into feature/label generation.

    Raw bars are cut at holdout_start first. Features and labels are then generated only
    inside the development partition. Rows whose label horizon would extend beyond the
    available development partition naturally receive NaN labels and are dropped.
    """
    registry=HoldoutRegistry(holdout_registry_path)
    policy=registry.load()
    cutoff=datetime.fromisoformat(policy['holdout_start']).astimezone(UTC)

    bars=pd.read_parquet(parquet_path)
    if 'timestamp' not in bars.columns:
        raise ValueError('dataset must contain timestamp')
    bars=bars.copy()
    bars['timestamp']=pd.to_datetime(bars['timestamp'],utc=True)
    assert_monotonic_time(bars)

    development=bars.loc[bars['timestamp'] < pd.Timestamp(cutoff)].copy()
    if development.empty:
        raise RuntimeError('no development rows exist before frozen holdout_start')
    if not (development['timestamp'] < pd.Timestamp(cutoff)).all():
        raise RuntimeError('holdout leakage: development partition crossed frozen holdout boundary')

    engine=feature_engine or FeatureEngine()
    featured=engine.transform(development)
    label_column=f'label__{label_contract.name}__{label_contract.label_version}'
    featured[label_column]=make_labels(featured['close'],label_contract)

    base_columns={'timestamp','open','high','low','close','tick_volume','spread','spread_points','real_volume'}
    feature_columns=tuple(c for c in featured.columns if c not in base_columns and c != label_column)
    assert_no_label_features(feature_columns)

    model_frame=featured.loc[featured[label_column].notna()].copy()
    if model_frame.empty:
        raise RuntimeError('no labeled development rows after feature/label construction')
    if not (model_frame['timestamp'] < pd.Timestamp(cutoff)).all():
        raise RuntimeError('holdout leakage: labeled development rows crossed frozen holdout boundary')

    return DevelopmentDataset(model_frame,feature_columns,label_column,cutoff)
