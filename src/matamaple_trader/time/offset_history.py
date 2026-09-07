from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from matamaple_trader.time.normalizer import BrokerOffsetObservation, ensure_utc


@dataclass(slots=True)
class BrokerOffsetHistory:
    """Append-only effective history of observed broker UTC offsets."""

    _observations: list[BrokerOffsetObservation] = field(default_factory=list)

    @property
    def observations(self) -> tuple[BrokerOffsetObservation, ...]:
        return tuple(self._observations)

    def record(self, observation: BrokerOffsetObservation) -> bool:
        observed_at = ensure_utc(observation.observed_at_utc)
        normalized = BrokerOffsetObservation(observed_at, observation.broker_server_time, observation.offset)
        if self._observations and observed_at <= self._observations[-1].observed_at_utc:
            raise ValueError("Offset observations must be appended in strictly increasing UTC time")
        changed = not self._observations or normalized.offset != self._observations[-1].offset
        self._observations.append(normalized)
        return changed

    def offset_at(self, timestamp_utc: datetime) -> timedelta:
        timestamp = ensure_utc(timestamp_utc)
        if not self._observations:
            raise LookupError("No broker offset observations recorded")
        keys = [o.observed_at_utc for o in self._observations]
        index = bisect_right(keys, timestamp) - 1
        if index < 0:
            raise LookupError("No broker offset observation exists at or before requested time")
        return self._observations[index].offset
