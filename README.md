# MATAMAPLE TRADER

Local-first Quant/ML **signal-only** platform for XM / MetaTrader 5.

## Safety contract

The system produces decision-support signals only. Human users decide whether to execute trades manually in MT5. Automatic order execution is forbidden; `mt5.order_send()` must not exist in the codebase.

## Build order status

- [x] 01 Foundation
- [x] 02 MT5 Connector — read-only; real Windows/XM terminal validation pending
- [x] 03 Time / DST Normalization
- [x] 04 Historical Collector — raw source order/duplicates preserved; gated chunked XM collection + manifests present; real XM history-depth/session validation pending
- [x] 05 Data Integrity Checks — duplicate/time reversal/OHLC/missing/spread/stale-tick/timeframe-alignment checks present; holiday/session calendar validation pending
- [x] 06 Broker Spec Integrity — persistent snapshots + critical drift/manual revalidation path present; real XM snapshots pending
- [x] 07 Real-Time Watcher — completed-bar + ATR/spread/structure/volatility triggers and stale-tick block present; real MT5 polling validation pending
- [x] 08 Shared Pipeline Skeleton
- [x] 09 Leakage Test Framework — embedded/suffixed target/future columns blocked
- [x] 10 Frozen Holdout Policy — hash-protected registry/gate implemented; a real research holdout must be frozen from actual historical data before production training
- [x] 11 Feature Engine
- [x] 12 Label Contracts
- [x] 13 Logistic Regression Baseline — deterministic scaling
- [x] 14 XGBoost — raw classifier margins + explicit multiclass mapping
- [x] 15 LightGBM — raw classifier margins + explicit multiclass mapping
- [x] 16 Purged Validation / CPCV
- [x] 17 Walk-Forward
- [x] 18 Probability Calibration
- [x] 19 Market Regime
- [x] 20 Cost / Slippage / Swap / Gap Engine
- [x] 21 Signal Engine
- [x] 22 Entry / SL / TP
- [x] 23 Signal Grade
- [x] 24 Grade Validation
- [x] 25 Portfolio Exposure Warning
- [x] 26 Backtest using Shared Pipeline
- [x] 27 Shadow Mode using Shared Pipeline — storage/runner implemented; real shadow observations pending
- [x] 28 Drift Monitoring
- [x] 29 Dashboard — optional Streamlit renderer
- [x] 30 XM Demo Validation framework — expected-vs-actual manual fill comparison implemented; real XM demo validation pending

## Critical gates still required before real use

1. Collect and integrity-check sufficient real XM historical data.
2. Review all dataset manifests and common coverage.
3. Explicitly freeze the real `research_holdout` once, before serious model selection/training.
4. Build development-only features/labels with the frozen holdout excluded before feature and label generation.
5. Run Purged CV/CPCV/Walk-Forward/frozen holdout without tuning on holdout outcomes.
6. Run real XM Shadow Mode and validate prediction, calibration, grade and drift behavior.
7. Validate expected versus manually executed XM Demo fills and refine cost assumptions from development/demo evidence only.
8. Do not enable automatic order execution; the project remains signal-only.

## Local setup (Windows 11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,data,ml,mt5,ui]"
pytest
```

## Read-only XM readiness check

With the XM MT5 terminal open and logged in:

```powershell
matamaple-xm-readiness --symbol EURUSD --timeframe M15 --start 2026-09-07T00:00:00+00:00 --end 2026-09-08T00:00:00+00:00 --min-bars 50
```

The command reads tick/spec/history only and prints a JSON report. Exit code `0` means the sample passed the configured readiness gates; exit code `2` means fail-safe/not ready. Use `--check-continuity` only after the requested symbol/session calendar has been validated, so normal weekend/holiday closures are not misclassified as missing bars.

## Gated real XM historical collection

After readiness succeeds, collect one or more symbols/timeframes through the read-only chunked collector:

```powershell
matamaple-xm-collect `
  --symbols EURUSD,GBPUSD,XAUUSD `
  --timeframes M15,H1 `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2026-09-01T00:00:00+00:00 `
  --chunk-days 30 `
  --min-bars 500
```

Each symbol/timeframe produces a manifest containing requested range, first/last available timestamp, bar count, quality failures and dataset SHA-256. The Parquet file is written only when the quality gate passes. A batch `data/historical/collection_manifest.json` summarizes all datasets. `--check-continuity` remains opt-in until XM session/holiday/DST behavior has been validated for the affected symbol/timeframe.

## Dataset review

Review the complete batch before proposing a holdout:

```powershell
matamaple-dataset-review --manifest data/historical/collection_manifest.json
```

Exit code `0` requires every listed dataset to be clean and to have a recorded Parquet artifact + SHA-256 plus a valid common history window. This command is read-only.

## Non-mutating research-holdout proposal

After dataset review succeeds:

```powershell
matamaple-holdout-proposal `
  --manifest data/historical/collection_manifest.json `
  --holdout-days 180 `
  --minimum-development-days 365
```

The command prints a proposed holdout start/end but **does not create or modify a holdout registry**.

## Explicit research-holdout freeze

Only after reviewing the proposal from real XM data, freeze the boundary once:

```powershell
matamaple-freeze-holdout `
  --manifest data/historical/collection_manifest.json `
  --holdout-days 180 `
  --minimum-development-days 365 `
  --policy-version research-holdout-v1 `
  --confirm-freeze
```

The default registry path is `data/validation/research_holdout.json`. It is hash-protected and cannot be overwritten through the registry API. **Do not run this command on mock/test manifests or simply because collection completed.** Once a real research holdout is frozen, never move or retune its boundary based on outcomes.

The development dataset builder reads the frozen registry, removes all rows at or after `holdout_start` first, and only then runs feature and label generation. Label-horizon rows that would require data beyond the development partition remain unavailable and are dropped rather than reading into the frozen holdout.

For CI or environments without MetaTrader 5/Streamlit, install with `pip install -e ".[dev,ml]"`. Install `.[data]` whenever Parquet historical storage is required.
