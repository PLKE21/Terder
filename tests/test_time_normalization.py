from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from matamaple_trader.time.normalizer import (
    BrokerOffsetObservation,
    TimeNormalizationError,
    broker_wall_time_to_utc,
    detect_server_offset,
    ensure_utc,
)
from matamaple_trader.time.offset_history import BrokerOffsetHistory


def test_naive_internal_timestamp_is_rejected() -> None:
    with pytest.raises(TimeNormalizationError):
        ensure_utc(datetime(2026, 1, 1, 12, 0))


def test_detects_broker_offset_without_hardcoding() -> None:
    obs = detect_server_offset(
        broker_server_time=datetime(2026, 1, 15, 14, 0, 3),
        observed_at_utc=datetime(2026, 1, 15, 12, 0, 1, tzinfo=UTC),
    )
    assert obs.offset == timedelta(hours=2)


def test_dst_like_server_shift_is_recorded_and_used_by_effective_time() -> None:
    history = BrokerOffsetHistory()
    winter = BrokerOffsetObservation(
        observed_at_utc=datetime(2026, 3, 28, 12, tzinfo=UTC),
        broker_server_time=datetime(2026, 3, 28, 14),
        offset=timedelta(hours=2),
    )
    summer = BrokerOffsetObservation(
        observed_at_utc=datetime(2026, 3, 29, 12, tzinfo=UTC),
        broker_server_time=datetime(2026, 3, 29, 15),
        offset=timedelta(hours=3),
    )
    assert history.record(winter) is True
    assert history.record(summer) is True
    assert history.offset_at(datetime(2026, 3, 28, 18, tzinfo=UTC)) == timedelta(hours=2)
    assert history.offset_at(datetime(2026, 3, 30, 18, tzinfo=UTC)) == timedelta(hours=3)
    assert broker_wall_time_to_utc(datetime(2026, 3, 30, 15), timedelta(hours=3)) == datetime(
        2026, 3, 30, 12, tzinfo=UTC
    )


def test_offset_history_rejects_time_reversal() -> None:
    history = BrokerOffsetHistory()
    history.record(
        BrokerOffsetObservation(
            observed_at_utc=datetime(2026, 4, 1, 12, tzinfo=UTC),
            broker_server_time=datetime(2026, 4, 1, 15),
            offset=timedelta(hours=3),
        )
    )
    with pytest.raises(ValueError):
        history.record(
            BrokerOffsetObservation(
                observed_at_utc=datetime(2026, 4, 1, 11, tzinfo=UTC),
                broker_server_time=datetime(2026, 4, 1, 14),
                offset=timedelta(hours=3),
            )
        )
