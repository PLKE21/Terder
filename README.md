# MATAMAPLE TRADER

Local-first Quant/ML **signal-only** platform for XM / MetaTrader 5.

## Safety contract

The system produces decision-support signals only. Human users decide whether to execute trades manually in MT5. Automatic order execution is forbidden; `mt5.order_send()` must not exist in the source implementation.

## V1 code status

The V1 code path is implemented through data readiness, gated collection, leakage controls, development-only model research, candidate lock, one-shot frozen holdout evaluation, shadow storage/readiness, drift monitoring, dashboard, manual XM demo fill validation and a final release-readiness gate.

This does **not** mean the system is validated for real-money use. Real XM historical collection, the real frozen research holdout, real shadow observations and manually executed XM demo fills still have to be produced on the user's Windows/XM environment before the final readiness command can pass.

## Build order status

- [x] 01 Foundation
- [x] 02 MT5 Connector — read-only; real Windows/XM terminal validation pending
- [x] 03 Time / DST Normalization
- [x] 04 Historical Collector — raw source order/duplicates preserved; gated chunked XM collection + manifests present
- [x] 05 Data Integrity Checks — duplicate/time reversal/OHLC/missing/spread/stale-tick/timeframe-alignment checks present
- [x] 06 Broker Spec Integrity — persistent snapshots + critical drift/manual revalidation path present
- [x] 07 Real-Time Watcher — completed-bar + ATR/spread/structure/volatility triggers and stale-tick block present
- [x] 08 Shared Pipeline Skeleton
- [x] 09 Leakage Test Framework
- [x] 10 Frozen Holdout Policy — hash-protected registry/gate
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
- [x] 27 Shadow Mode using Shared Pipeline — append-only storage + payload hashes
- [x] 28 Drift Monitoring
- [x] 29 Dashboard
- [x] 30 XM Demo Validation framework — manual fills only
- [x] Candidate lock before holdout evaluation
- [x] One-shot frozen holdout evaluation with development-only OOF calibration
- [x] Shadow/demo/release readiness gates

## Required real validation sequence

1. Run read-only XM readiness.
2. Collect sufficient real XM history and pass integrity review.
3. Propose and explicitly freeze the real research holdout once.
4. Run development-only model research. Do not inspect holdout outcomes.
5. Lock the selected candidate/config.
6. Evaluate the frozen holdout once. Do not retune from its outcome.
7. Run real XM Shadow Mode and attach outcomes later.
8. Record manually executed XM Demo fills and summarize expected-vs-actual costs.
9. Run final release readiness. Until it passes, treat the system as research/shadow/demo only.
10. Automatic order execution remains forbidden in all cases.

## Local setup (Windows 11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,data,ml,mt5,ui]"
pytest
```

## 1. Read-only XM readiness

```powershell
matamaple-xm-readiness --symbol EURUSD --timeframe M15 --start 2026-09-07T00:00:00+00:00 --end 2026-09-08T00:00:00+00:00 --min-bars 50
```

## 2. Gated historical collection

```powershell
matamaple-xm-collect `
  --symbols EURUSD,GBPUSD,XAUUSD `
  --timeframes M15,H1 `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2026-09-01T00:00:00+00:00 `
  --chunk-days 30 `
  --min-bars 500
```

## 3. Dataset review and holdout freeze

```powershell
matamaple-dataset-review --manifest data/historical/collection_manifest.json

matamaple-holdout-proposal `
  --manifest data/historical/collection_manifest.json `
  --holdout-days 180 `
  --minimum-development-days 365

matamaple-freeze-holdout `
  --manifest data/historical/collection_manifest.json `
  --holdout-days 180 `
  --minimum-development-days 365 `
  --policy-version research-holdout-v1 `
  --confirm-freeze
```

The frozen registry is hash-protected and cannot be overwritten through the registry API. Never move the boundary based on outcomes.

## 4. Development-only experiment

```powershell
matamaple-development-experiment `
  --parquet data/historical/EURUSD/M15.parquet `
  --holdout-registry data/validation/research_holdout.json `
  --horizon-bars 4 `
  --purged-splits 5 `
  --cpcv-groups 6 `
  --cpcv-test-groups 2 `
  --wf-min-train-bars 2000 `
  --wf-test-bars 250 `
  --output artifacts/EURUSD_M15_development_experiment.json
```

The experiment uses development data only, Purged CV/CPCV/expanding Walk-Forward and OOF calibration. There is no shuffled split option.

## 5. Lock candidate before holdout

```powershell
matamaple-lock-candidate `
  --development-report artifacts/EURUSD_M15_development_experiment.json `
  --feature-version features-v1 `
  --pipeline-version v0.1.0 `
  --horizon-bars 4 `
  --confirm-lock
```

The candidate lock records model choice, label/version, feature version, calibration method, validation parameters, pipeline version and SHA-256 of the development report. It is immutable through the registry API.

## 6. Evaluate frozen holdout once

```powershell
matamaple-evaluate-holdout `
  --parquet data/historical/EURUSD/M15.parquet `
  --holdout-registry data/validation/research_holdout.json `
  --candidate-lock data/validation/candidate_lock.json `
  --output artifacts/frozen_holdout_evaluation.json `
  --confirm-one-shot
```

Calibration is fitted from development OOF predictions only. The output is hash-protected, marks `tuning_allowed_after_evaluation: false`, and an existing report is never overwritten.

## 7. Shadow + XM Demo evidence

Shadow predictions remain append-only and outcomes are attached separately. Shadow prediction/outcome payloads carry SHA-256 hashes, and SQLite foreign-key checking is enabled on every connection.

XM Demo validation compares **manually executed** fills with expected fills. No order-placement API exists in the demo validator.

## 8. Final release-readiness gate

```powershell
matamaple-release-readiness `
  --holdout-report artifacts/frozen_holdout_evaluation.json `
  --shadow-db data/shadow/shadow.sqlite `
  --demo-summary artifacts/xm_demo_summary.json `
  --min-shadow-predictions 200 `
  --min-shadow-outcome-coverage 0.8 `
  --min-manual-fills 30 `
  --max-mean-abs-slippage-points 5
```

The report distinguishes `code_path_complete` from `validated_for_real_use`. It always reports `auto_trading_enabled: false`. Missing real evidence fails safe with exit code `2`.

## Current limitations before real release-candidate status

Real XM session/holiday/DST behavior must still be validated per affected symbol/timeframe. Real broker-spec snapshots must be collected. Current V1 experiment/one-shot holdout calibration is binary-classification only; multiclass probability calibration must be hardened separately before production use of `P(BUY)/P(NEUTRAL)/P(SELL)`. Real shadow observations and manual XM Demo fill evidence are not present in the repository and cannot be fabricated by CI.

For CI or environments without MetaTrader 5/Streamlit, install with `pip install -e ".[dev,ml]"`. Install `.[data]` whenever Parquet historical storage is required.
