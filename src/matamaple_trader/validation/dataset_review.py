from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetReviewItem:
    symbol: str
    timeframe: str
    quality_ok: bool
    bar_count: int
    first_available_timestamp: datetime | None
    last_available_timestamp: datetime | None
    dataset_sha256: str | None
    parquet_path: str | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetReviewReport:
    dataset_count: int
    clean_dataset_count: int
    failed_dataset_count: int
    common_start: datetime | None
    common_end: datetime | None
    ready_for_holdout_proposal: bool
    reasons: tuple[str, ...]
    items: tuple[DatasetReviewItem, ...]


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('manifest timestamps must be timezone-aware')
    return parsed.astimezone(UTC)


def review_collection_manifest(path: str | Path) -> DatasetReviewReport:
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    raw_items=payload.get('datasets', [])
    items=[]
    reasons=[]
    for raw in raw_items:
        item=DatasetReviewItem(
            symbol=str(raw.get('symbol','')),
            timeframe=str(raw.get('timeframe','')),
            quality_ok=bool(raw.get('quality_ok',False)),
            bar_count=int(raw.get('bar_count',0)),
            first_available_timestamp=_dt(raw.get('first_available_timestamp')),
            last_available_timestamp=_dt(raw.get('last_available_timestamp')),
            dataset_sha256=raw.get('dataset_sha256'),
            parquet_path=raw.get('parquet_path'),
            failures=tuple(raw.get('quality_failures',()) or ()),
        )
        items.append(item)
        if not item.symbol or not item.timeframe:
            reasons.append('invalid_dataset_identity')
        if item.quality_ok and (not item.dataset_sha256 or not item.parquet_path):
            reasons.append(f'missing_dataset_artifact:{item.symbol}:{item.timeframe}')
        if not item.quality_ok:
            reasons.append(f'quality_failed:{item.symbol}:{item.timeframe}')

    clean=[x for x in items if x.quality_ok and x.first_available_timestamp and x.last_available_timestamp]
    if not items:
        reasons.append('no_datasets')
    if len(clean) != len(items):
        reasons.append('not_all_datasets_clean')
    common_start=max((x.first_available_timestamp for x in clean), default=None)
    common_end=min((x.last_available_timestamp for x in clean), default=None)
    if common_start and common_end and common_end <= common_start:
        reasons.append('no_common_history_window')
    ready=bool(items) and not reasons
    return DatasetReviewReport(
        dataset_count=len(items),
        clean_dataset_count=sum(1 for x in items if x.quality_ok),
        failed_dataset_count=sum(1 for x in items if not x.quality_ok),
        common_start=common_start,
        common_end=common_end,
        ready_for_holdout_proposal=ready,
        reasons=tuple(dict.fromkeys(reasons)),
        items=tuple(items),
    )
