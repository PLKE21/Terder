from __future__ import annotations

from dataclasses import dataclass

from matamaple_trader.domain import Signal

INDEX_TOKENS = ("US30", "US500", "SPX", "NAS", "USTEC", "GER40", "DE40", "UK100", "JP225")


@dataclass(frozen=True, slots=True)
class SignalExposure:
    symbol: str
    signal: Signal
    group: str | None = None


def infer_exposure_group(symbol: str) -> str:
    upper = symbol.upper()
    if upper.startswith(("XAU", "XAG")):
        return "METALS"
    if any(token in upper for token in INDEX_TOKENS):
        return "INDICES"
    if "USD" in upper:
        return "USD"
    return "RELATED"


def exposure_warning(exposures: list[SignalExposure], *, threshold: int = 3) -> str | None:
    if threshold < 2:
        raise ValueError("threshold must be >= 2")
    counts: dict[tuple[str, Signal], int] = {}
    for item in exposures:
        if item.signal not in {Signal.BUY, Signal.SELL}:
            continue
        group = item.group or infer_exposure_group(item.symbol)
        key = (group, item.signal)
        counts[key] = counts.get(key, 0) + 1
    warnings = [
        f"informational exposure warning: {count} {signal.value} signals in {group} group"
        for (group, signal), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1].value))
        if count >= threshold
    ]
    return "; ".join(warnings) if warnings else None
