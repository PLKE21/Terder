from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetCoverage:
    symbol: str
    timeframe: str
    first_available_timestamp: datetime
    last_available_timestamp: datetime
    bar_count: int
    quality_ok: bool


@dataclass(frozen=True, slots=True)
class HoldoutProposal:
    ready_to_freeze: bool
    common_start: datetime | None
    common_end: datetime | None
    common_coverage_days: int
    proposed_holdout_start: datetime | None
    proposed_holdout_end: datetime | None
    holdout_days: int
    minimum_development_days: int
    reasons: tuple[str, ...]


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("manifest timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def load_coverages(collection_manifest: str | Path) -> list[DatasetCoverage]:
    payload = json.loads(Path(collection_manifest).read_text(encoding="utf-8"))
    datasets = payload.get("datasets", [])
    out: list[DatasetCoverage] = []
    for item in datasets:
        first = item.get("first_available_timestamp")
        last = item.get("last_available_timestamp")
        if first is None or last is None:
            continue
        out.append(DatasetCoverage(
            symbol=str(item["symbol"]),
            timeframe=str(item["timeframe"]),
            first_available_timestamp=_dt(first),
            last_available_timestamp=_dt(last),
            bar_count=int(item.get("bar_count", 0)),
            quality_ok=bool(item.get("quality_ok", False)),
        ))
    return out


def propose_holdout(
    coverages: list[DatasetCoverage],
    *,
    holdout_days: int = 180,
    minimum_development_days: int = 365,
) -> HoldoutProposal:
    if holdout_days <= 0 or minimum_development_days <= 0:
        raise ValueError("holdout_days and minimum_development_days must be positive")
    reasons: list[str] = []
    if not coverages:
        reasons.append("no_usable_dataset_coverage")
        return HoldoutProposal(False, None, None, 0, None, None, holdout_days, minimum_development_days, tuple(reasons))
    bad = [f"{x.symbol}:{x.timeframe}" for x in coverages if not x.quality_ok]
    if bad:
        reasons.append("quality_gate_failed:" + ",".join(sorted(bad)))

    common_start = max(x.first_available_timestamp for x in coverages)
    common_end = min(x.last_available_timestamp for x in coverages)
    coverage_days = max(0, (common_end - common_start).days)
    required = minimum_development_days + holdout_days
    if common_end <= common_start:
        reasons.append("no_common_history_window")
    elif coverage_days < required:
        reasons.append(f"insufficient_common_history:{coverage_days}<{required}_days")

    ready = not reasons
    proposed_start = common_end - timedelta(days=holdout_days) if ready else None
    proposed_end = common_end if ready else None
    return HoldoutProposal(
        ready_to_freeze=ready,
        common_start=common_start,
        common_end=common_end,
        common_coverage_days=coverage_days,
        proposed_holdout_start=proposed_start,
        proposed_holdout_end=proposed_end,
        holdout_days=holdout_days,
        minimum_development_days=minimum_development_days,
        reasons=tuple(reasons),
    )
