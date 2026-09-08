from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path

import pandas as pd

from matamaple_trader.data.integrity import DataQualityReport, validate_bars


@dataclass(frozen=True, slots=True)
class CollectionRequest:
    symbol: str
    timeframe_code: int
    timeframe_name: str
    start_utc: datetime
    end_utc: datetime
    expected_delta: pd.Timedelta | None
    min_bars: int = 100
    max_spread_points: int | None = None
    chunk_days: int = 30

    def __post_init__(self) -> None:
        if self.start_utc.tzinfo is None or self.end_utc.tzinfo is None:
            raise ValueError("collection timestamps must be timezone-aware")
        if self.end_utc <= self.start_utc:
            raise ValueError("end_utc must be after start_utc")
        if self.min_bars <= 0 or self.chunk_days <= 0:
            raise ValueError("min_bars and chunk_days must be positive")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    symbol: str
    timeframe: str
    requested_start: datetime
    requested_end: datetime
    first_available_timestamp: datetime | None
    last_available_timestamp: datetime | None
    bar_count: int
    source: str
    quality_ok: bool
    quality_failures: tuple[str, ...]
    missing_ranges: tuple[str, ...]
    dataset_sha256: str | None
    parquet_path: str | None
    collected_at: datetime


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class XMHistoricalCollectionPipeline:
    """Read-only, integrity-gated historical collection.

    Raw chunks are validated before canonicalization. A Parquet dataset is written only
    when the combined raw data passes the configured quality gate.
    """

    def __init__(self, connector, root: str | Path = "data/historical") -> None:
        self.connector = connector
        self.root = Path(root)

    def collect(self, request: CollectionRequest) -> tuple[pd.DataFrame, DataQualityReport, DatasetManifest]:
        start = request.start_utc.astimezone(UTC)
        end = request.end_utc.astimezone(UTC)
        chunk = timedelta(days=request.chunk_days)
        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + chunk, end)
            frame = self.connector.get_rates(request.symbol, request.timeframe_code, cursor, chunk_end).copy()
            if not frame.empty:
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frames.append(frame)
            cursor = chunk_end

        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        report = validate_bars(
            raw,
            request.expected_delta,
            min_bars=request.min_bars,
            max_spread_points=request.max_spread_points,
        )

        canonical = _canonical_frame(raw)
        first = None if canonical.empty else canonical["timestamp"].iloc[0].to_pydatetime()
        last = None if canonical.empty else canonical["timestamp"].iloc[-1].to_pydatetime()
        parquet_path: str | None = None
        sha: str | None = None
        if report.ok:
            path = self.root / request.symbol / f"{request.timeframe_name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            canonical.to_parquet(path, index=False)
            parquet_path = str(path)
            sha = _frame_hash(canonical)

        manifest = DatasetManifest(
            symbol=request.symbol,
            timeframe=request.timeframe_name,
            requested_start=start,
            requested_end=end,
            first_available_timestamp=first,
            last_available_timestamp=last,
            bar_count=len(canonical),
            source=self.connector.__class__.__name__,
            quality_ok=report.ok,
            quality_failures=report.failures,
            missing_ranges=report.missing_ranges,
            dataset_sha256=sha,
            parquet_path=parquet_path,
            collected_at=datetime.now(UTC),
        )
        self.write_manifest(manifest)
        return canonical, report, manifest

    def write_manifest(self, manifest: DatasetManifest) -> Path:
        path = self.root / manifest.symbol / f"{manifest.timeframe}.manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(manifest)
        for key in ("requested_start", "requested_end", "first_available_timestamp", "last_available_timestamp", "collected_at"):
            value = payload[key]
            payload[key] = value.isoformat() if value is not None else None
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
