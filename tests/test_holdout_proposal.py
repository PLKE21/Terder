from datetime import UTC, datetime, timedelta

from matamaple_trader.validation.holdout_proposal import DatasetCoverage, propose_holdout


def cov(symbol='EURUSD',tf='M15',start=None,end=None,quality=True):
    start=start or datetime(2024,1,1,tzinfo=UTC)
    end=end or datetime(2026,1,1,tzinfo=UTC)
    return DatasetCoverage(symbol,tf,start,end,1000,quality)


def test_holdout_proposal_uses_common_window_only():
    items=[
        cov('EURUSD','M15',datetime(2024,1,1,tzinfo=UTC),datetime(2026,1,1,tzinfo=UTC)),
        cov('XAUUSD','H1',datetime(2024,3,1,tzinfo=UTC),datetime(2025,12,1,tzinfo=UTC)),
    ]
    proposal=propose_holdout(items,holdout_days=180,minimum_development_days=365)
    assert proposal.ready_to_freeze
    assert proposal.common_start==datetime(2024,3,1,tzinfo=UTC)
    assert proposal.common_end==datetime(2025,12,1,tzinfo=UTC)
    assert proposal.proposed_holdout_end==proposal.common_end
    assert proposal.proposed_holdout_start==proposal.common_end-timedelta(days=180)


def test_holdout_proposal_rejects_quality_failure():
    proposal=propose_holdout([cov(quality=False)])
    assert not proposal.ready_to_freeze
    assert any(reason.startswith('quality_gate_failed:') for reason in proposal.reasons)
    assert proposal.proposed_holdout_start is None


def test_holdout_proposal_rejects_short_common_history():
    start=datetime(2025,1,1,tzinfo=UTC)
    end=datetime(2025,10,1,tzinfo=UTC)
    proposal=propose_holdout([cov(start=start,end=end)],holdout_days=180,minimum_development_days=365)
    assert not proposal.ready_to_freeze
    assert any(reason.startswith('insufficient_common_history:') for reason in proposal.reasons)
