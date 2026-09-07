# MATAMAPLE TRADER

Local-first Quant/ML **signal-only** platform for XM / MetaTrader 5.

## Safety contract

The system produces decision-support signals only. Human users decide whether to execute trades manually in MT5. Automatic order execution is forbidden; `mt5.order_send()` must not exist in the codebase.

## Build order status

- [x] 01 Foundation
- [x] 02 MT5 Connector (read-only market-data/spec connector; real terminal validation still required on Windows)
- [x] 03 Time / DST Normalization
- [ ] 04 Historical Collector
- [ ] 05 Data Integrity Checks
- [ ] 06 Broker Spec Integrity
- [ ] 07 Real-Time Watcher
- [ ] 08 Shared Pipeline Skeleton
- [ ] 09 Leakage Test Framework
- [ ] 10 Frozen Holdout Policy
- [ ] 11+ Remaining phases

> Training is intentionally blocked until phases 08, 09, and 10 exist and pass their gates.

## Local setup (Windows 11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,mt5]"
pytest
```

For environments without MetaTrader 5 installed, install with `pip install -e ".[dev]"`; tests use dependency injection/mocks for the connector.
