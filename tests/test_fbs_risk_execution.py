from datetime import UTC, datetime

from matamaple_trader.domain import OperationalState, Signal, SignalResult
from matamaple_trader.execution import ExecutionContext, FBSExecutionGuard
from matamaple_trader.risk import AccountRiskState, RiskConfig, RiskEngine


def signal_result(signal=Signal.BUY):
    return SignalResult(
        timestamp=datetime(2026, 9, 9, tzinfo=UTC),
        symbol='EURUSD',
        signal=signal,
        raw_score=0.8,
        calibrated_probability=0.8,
        regime='TREND',
        entry_low=1.1,
        entry_high=1.1002,
        stop_loss=1.095,
        tp1=1.105,
        tp2=1.11,
        tp3=1.115,
        risk_reward=2.0,
        grade='A',
        spread_state='NORMAL',
        portfolio_warning=None,
        pipeline_version='v0.3.0-fbs-ai',
        operational_state=OperationalState.ACTIVE,
    )


def clean_account(**overrides):
    data=dict(equity=10000.0,balance=10000.0,daily_pnl=0.0,peak_equity=10000.0,free_margin=9000.0,open_positions=0)
    data.update(overrides)
    return AccountRiskState(**data)


def clean_execution(**overrides):
    data=dict(
        spread_points=10.0,
        max_spread_points=25.0,
        stop_distance_points=500.0,
        min_stop_distance_points=20.0,
        freeze_distance_points=10.0,
        estimated_cost=2.0,
        max_estimated_cost=5.0,
        terminal_trade_allowed=True,
        symbol_trade_allowed=True,
    )
    data.update(overrides)
    return ExecutionContext(**data)


def test_clean_trade_passes_risk_and_execution_guard():
    risk=RiskEngine().evaluate(Signal.BUY,clean_account())
    execution=FBSExecutionGuard().evaluate(signal_result(),risk,clean_execution())
    assert risk.allowed
    assert risk.max_risk_amount == 100.0
    assert execution.allowed


def test_daily_loss_limit_blocks_trade():
    risk=RiskEngine(RiskConfig(max_daily_loss_pct=3.0)).evaluate(Signal.BUY,clean_account(daily_pnl=-350.0))
    execution=FBSExecutionGuard().evaluate(signal_result(),risk,clean_execution())
    assert not risk.allowed
    assert risk.reason == 'daily_loss_limit'
    assert not execution.allowed
    assert execution.reason.startswith('risk_block:')


def test_spread_limit_blocks_trade_after_risk_passes():
    risk=RiskEngine().evaluate(Signal.BUY,clean_account())
    execution=FBSExecutionGuard().evaluate(signal_result(),risk,clean_execution(spread_points=30.0))
    assert risk.allowed
    assert not execution.allowed
    assert execution.reason == 'spread_limit'


def test_stop_or_freeze_level_blocks_trade():
    risk=RiskEngine().evaluate(Signal.BUY,clean_account())
    execution=FBSExecutionGuard().evaluate(
        signal_result(),risk,clean_execution(stop_distance_points=15.0,min_stop_distance_points=20.0,freeze_distance_points=25.0)
    )
    assert not execution.allowed
    assert execution.reason == 'stop_or_freeze_level'


def test_non_actionable_signal_cannot_reach_execution():
    risk=RiskEngine().evaluate(Signal.WAIT,clean_account())
    execution=FBSExecutionGuard().evaluate(signal_result(Signal.WAIT),risk,clean_execution())
    assert not risk.allowed
    assert not execution.allowed
