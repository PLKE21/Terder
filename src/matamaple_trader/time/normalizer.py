from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True, slots=True)
class BrokerOffsetObservation:
    observed_at_utc: datetime
    broker_server_time: datetime
    offset: timedelta


class TimeNormalizationError(ValueError):
    pass


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise TimeNormalizationError("Naive datetime is forbidden; timezone-aware input is required")
    return value.astimezone(UTC)


def detect_server_offset(*, broker_server_time: datetime, observed_at_utc: datetime) -> BrokerOffsetObservation:
    """Infer broker-server offset without hard-coding GMT+2/GMT+3."""

    utc_now = ensure_utc(observed_at_utc)
    broker_wall = broker_server_time.replace(tzinfo=None)
    utc_wall = utc_now.replace(tzinfo=None)
    raw_seconds = (broker_wall - utc_wall).total_seconds()
    rounded_minutes = round(raw_seconds / 60.0)
    offset = timedelta(minutes=rounded_minutes)
    if abs(offset) > timedelta(hours=14):
        raise TimeNormalizationError(f"Implausible broker UTC offset: {offset}")
    return BrokerOffsetObservation(observed_at_utc=utc_now, broker_server_time=broker_wall, offset=offset)


def broker_wall_time_to_utc(broker_server_time: datetime, offset: timedelta) -> datetime:
    """Convert broker wall-clock time using the offset valid for that observation period."""

    wall = broker_server_time.replace(tzinfo=None)
    return (wall - offset).replace(tzinfo=UTC)
