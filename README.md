# MATAMAPLE TRADER

MATAMAPLE TRADER is a local-first Quant/ML trading platform targeting **FBS MetaTrader 5**.

Current engineering milestone: **v0.4.0 FBS Demo-Ready**.

The system can collect FBS MT5 data, run the shared quantitative pipeline, apply a local Qwen/Ollama reviewer, enforce deterministic account risk and execution gates, perform broker-side margin/PnL preflight calculations, and submit orders only to a verified MT5 **demo account**. Live trading remains hard-disabled.

## Runtime architecture

```text
FBS MT5 market data
        |
        v
Shared Quant Pipeline
        |
        v
Local Qwen Reviewer (Ollama)
        |
        v
AI Circuit Breaker / Fail-Closed Gate
        |
        v
Deterministic Risk Engine
        |
        v
FBS Execution Guard
        |
        v
Auto Lot + MT5 order_calc_margin/order_calc_profit
        |
        v
order_check
        |
        v
DRY_RUN or verified FBS DEMO order_send
        |
        v
Runtime + Trade Journals
```

AI is reviewer-only. It cannot create BUY/SELL direction, change lot size, SL/TP, bypass risk controls, or enable live trading.

## Safety contract

- Internal time is UTC. Broker server offsets/DST must be normalized; no hardcoded GMT offset.
- Backtest and runtime quantitative logic must share the same feature/signal/Entry/SL/TP code path.
- Regime thresholds are tuned only on development/training/validation data, never final holdout.
- Final holdout remains one-shot and must not be used for retuning.
- Risk, spread, stop/freeze, cost, margin and account-state checks are deterministic.
- Ollama failures, excessive latency, malformed reviews and an open AI circuit fail closed to WAIT.
- `DRY_RUN` is the default execution mode.
- Demo submission requires an MT5 account whose trade mode is DEMO and a successful `order_check`.
- LIVE trading is hard-disabled by both runtime configuration and the demo sender.

## Windows setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,data,ml,mt5,ui]"
pytest
```

Install Ollama separately and load the default local reviewer model:

```powershell
ollama run qwen3:4b
```

The default Ollama endpoint is loopback-only: `http://127.0.0.1:11434`. Runtime configuration rejects non-loopback hosts and embedded URL credentials.

## 1. FBS MT5 readiness

Open FBS MetaTrader 5 and log in to the intended account first.

```powershell
matamaple-fbs-readiness `
  --symbol EURUSD `
  --timeframe M15 `
  --start 2026-09-07T00:00:00+00:00 `
  --end 2026-09-08T00:00:00+00:00 `
  --min-bars 50
```

## 2. Collect FBS historical data

```powershell
matamaple-fbs-collect `
  --symbols EURUSD,GBPUSD,XAUUSD `
  --timeframes M15,H1 `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2026-09-01T00:00:00+00:00 `
  --chunk-days 30 `
  --min-bars 500
```

Default historical root: `data/fbs/historical`.

## 3. Research / holdout workflow

Review data, freeze the holdout once, develop only on development data, lock the candidate, then evaluate the final holdout once.

```powershell
matamaple-dataset-review --manifest data/fbs/historical/collection_manifest.json

matamaple-holdout-proposal `
  --manifest data/fbs/historical/collection_manifest.json `
  --holdout-days 180 `
  --minimum-development-days 365

matamaple-freeze-holdout `
  --manifest data/fbs/historical/collection_manifest.json `
  --holdout-days 180 `
  --minimum-development-days 365 `
  --policy-version research-holdout-v1 `
  --confirm-freeze
```

Development experiments use purged validation/CPCV/walk-forward and development-only calibration. Do not inspect or reuse final-holdout outcomes for tuning.

## 4. FBS execution path

The production-facing demo path is:

1. Quant pipeline produces BUY/SELL candidate.
2. Qwen reviews only that candidate.
3. AI circuit breaker rejects calls after repeated local-model failures until cooldown.
4. Risk Engine computes the maximum risk budget.
5. Auto-lot sizing respects broker `volume_min`, `volume_max` and `volume_step`.
6. FBS Execution Guard checks spread, stop/freeze distance, cost and trade permissions.
7. `MT5Connector.order_calc_margin()` and `order_calc_profit()` validate margin and stop-loss risk.
8. Demo sender requires `order_check` success.
9. Only a verified DEMO account can reach `order_send`.
10. Every runtime decision and demo send is written to append-only JSONL journals.

The market-data connector itself intentionally has **no `order_send` method**.

## 5. Demo readiness audit

After collecting real FBS demo/forward-test evidence, prepare a metrics JSON file:

```json
{
  "runtime_cycles": 250,
  "actionable_candidates": 45,
  "demo_orders_submitted": 42,
  "order_send_failures": 0,
  "order_check_rejections": 0,
  "ai_failures": 1,
  "ai_latency_violations": 0,
  "max_observed_drawdown_pct": 2.4,
  "max_daily_loss_pct": 1.1,
  "journal_parse_errors": 0
}
```

Then run:

```powershell
matamaple-demo-readiness --metrics artifacts/fbs_demo_metrics.json
```

The default engineering gate requires at least 100 runtime cycles, at least 20 actionable candidates, no journal-integrity errors, controlled order/AI failure rates, drawdown <= 10%, and daily loss <= 3%. A passing result means **ready for extended demo/forward testing only**. It does not enable or authorize live trading.

## Current status

Implemented and covered by CI:

- FBS-first data collection and readiness
- UTC/time normalization and integrity checks
- shared quantitative pipeline, backtest and shadow foundations
- leakage/frozen-holdout controls
- model research, calibration and regime/grade validation foundations
- cost/slippage/swap/gap simulation
- local Qwen/Ollama reviewer
- fail-closed AI circuit breaker
- deterministic account Risk Engine
- FBS Execution Guard
- broker-compatible auto lot sizing
- MT5 margin/PnL preflight calculations
- `order_check` protected FBS demo sender
- real-account rejection for demo sender
- kill switch and broker retcode handling
- runtime and trade JSONL journals
- deterministic Demo Readiness Gate

Still requires user-generated evidence on the Windows/FBS environment before calling the strategy validated: real FBS historical data, final frozen research evaluation, sufficient shadow observations, and sufficient FBS demo forward-test evidence. These observations cannot be fabricated by CI.

## Live trading

**LIVE trading is not enabled in v0.4.0.** The repository is intentionally locked to DRY_RUN/DEMO while real FBS demo evidence is collected and audited. Passing software tests or the demo readiness gate is not a guarantee of profitability or real-money safety.
