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
2. Freeze the real `research_holdout` before serious model selection/training.
3. Run Purged CV/CPCV/Walk-Forward/frozen holdout without tuning on holdout outcomes.
4. Run real XM Shadow Mode and validate prediction, calibration, grade and drift behavior.
5. Validate expected versus manually executed XM Demo fills and refine cost assumptions from development/demo evidence only.
6. Do not enable automatic order execution; the project remains signal-only.

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

**Do not freeze `research_holdout` merely because collection completed.** First review history depth and integrity across all required datasets; only then choose a fixed holdout boundary and freeze it before serious model selection.

For CI or environments without MetaTrader 5/Streamlit, install with `pip install -e ".[dev,ml]"`. Install `.[data]` whenever Parquet historical storage is required.
