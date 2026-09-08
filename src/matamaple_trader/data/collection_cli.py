from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import pandas as pd

from matamaple_trader.adapters.mt5_connector import MT5Connector
from matamaple_trader.data.xm_collection import CollectionRequest, XMHistoricalCollectionPipeline

TIMEFRAME_DELTAS = {
    "M1": pd.Timedelta("1min"),
    "M5": pd.Timedelta("5min"),
    "M15": pd.Timedelta("15min"),
    "M30": pd.Timedelta("30min"),
    "H1": pd.Timedelta("1h"),
    "H4": pd.Timedelta("4h"),
    "D1": pd.Timedelta("1d"),
}


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include timezone")
    return dt.astimezone(UTC)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _jsonable_manifest(manifest) -> dict:
    payload = asdict(manifest)
    for key in ("requested_start", "requested_end", "first_available_timestamp", "last_available_timestamp", "collected_at"):
        value = payload[key]
        payload[key] = value.isoformat() if value is not None else None
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only XM historical collector. Writes only datasets that pass integrity gates.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. EURUSD,XAUUSD")
    parser.add_argument("--timeframes", required=True, help="Comma-separated: M1,M5,M15,M30,H1,H4,D1")
    parser.add_argument("--start", required=True, type=_parse_utc)
    parser.add_argument("--end", required=True, type=_parse_utc)
    parser.add_argument("--root", default="data/historical")
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--min-bars", type=int, default=100)
    parser.add_argument("--max-spread-points", type=int, default=None)
    parser.add_argument("--check-continuity", action="store_true", help="Enable exact bar-gap checks only after validating XM session/holiday behavior for the period.")
    args = parser.parse_args(argv)

    symbols = _csv(args.symbols)
    timeframes = _csv(args.timeframes)
    unknown = [tf for tf in timeframes if tf not in TIMEFRAME_DELTAS]
    if not symbols:
        parser.error("at least one symbol is required")
    if not timeframes or unknown:
        parser.error(f"unsupported timeframe(s): {unknown}")

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        parser.error("MetaTrader5 is not installed. Install with pip install -e '.[mt5]'")
        raise exc

    connector = MT5Connector(mt5)
    pipeline = XMHistoricalCollectionPipeline(connector, args.root)
    results: list[dict] = []
    all_ok = True
    try:
        connector.connect()
        for symbol in symbols:
            for timeframe in timeframes:
                request = CollectionRequest(
                    symbol=symbol,
                    timeframe_code=getattr(mt5, f"TIMEFRAME_{timeframe}"),
                    timeframe_name=timeframe,
                    start_utc=args.start,
                    end_utc=args.end,
                    expected_delta=TIMEFRAME_DELTAS[timeframe] if args.check_continuity else None,
                    min_bars=args.min_bars,
                    max_spread_points=args.max_spread_points,
                    chunk_days=args.chunk_days,
                )
                _, _, manifest = pipeline.collect(request)
                results.append(_jsonable_manifest(manifest))
                all_ok = all_ok and manifest.quality_ok
    finally:
        connector.close()

    index = {
        "generated_at": datetime.now(UTC).isoformat(),
        "all_quality_ok": all_ok,
        "datasets": results,
    }
    index_path = Path(args.root) / "collection_manifest.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
