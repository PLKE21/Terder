# MATAMAPLE TRADER

Local-first Quant/ML **signal-only** platform for XM / MetaTrader 5.

## Safety contract

The system produces decision-support signals only. Human users decide whether to execute trades manually in MT5. Automatic order execution is forbidden; `mt5.order_send()` must not exist in the codebase.

## Build order status

- [x] 01 Foundation
- [x] 02 MT5 Connector — read-only; real Windows/XM terminal validation pending
- [x] 03 Time / DST Normalization
- [x] 04 Historical Collector — implementation present; production history/session hardening still required
- [x] 05 Data Integrity Checks — implementation present; live/session-aware hardening still required
- [x] 06 Broker Spec Integrity — critical drift/manual revalidation path present; real XM snapshots pending
- [x] 07 Real-Time Watcher — measurable trigger logic present; real MT5 polling validation pending
- [x] 08 Shared Pipeline Skeleton
- [x] 09 Leakage Test Framework
- [x] 10 Frozen Holdout Policy — policy/gate implemented; a real research holdout must be frozen from actual historical data before production training
- [x] 11 Feature Engine
- [x] 12 Label Contracts
- [x] 13 Logistic Regression Baseline
- [x] 14 XGBoost
- [x] 15 LightGBM
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
pip install -e ".[dev,ml,mt5,ui]"
pytest
```

For CI or environments without MetaTrader 5/Streamlit, install with `pip install -e ".[dev,ml]"`.
