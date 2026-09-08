from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3

from matamaple_trader.domain import SignalResult


@dataclass(frozen=True, slots=True)
class ShadowPrediction:
    prediction_id: int
    timestamp: str
    symbol: str
    signal: str
    pipeline_version: str


class ShadowStore:
    """Append-only shadow predictions; outcomes are attached separately later."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, symbol TEXT NOT NULL, signal TEXT NOT NULL, pipeline_version TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(timestamp, symbol, pipeline_version))")
            con.execute("CREATE TABLE IF NOT EXISTS outcomes (prediction_id INTEGER PRIMARY KEY, payload TEXT NOT NULL, FOREIGN KEY(prediction_id) REFERENCES predictions(id))")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def append_prediction(self, result: SignalResult) -> int:
        payload = {
            "raw_score": result.raw_score,
            "calibrated_probability": result.calibrated_probability,
            "regime": result.regime,
            "entry_low": result.entry_low,
            "entry_high": result.entry_high,
            "stop_loss": result.stop_loss,
            "tp1": result.tp1,
            "tp2": result.tp2,
            "tp3": result.tp3,
            "risk_reward": result.risk_reward,
            "grade": result.grade,
            "spread_state": result.spread_state,
            "portfolio_warning": result.portfolio_warning,
            "operational_state": result.operational_state.value,
        }
        with self._connect() as con:
            try:
                cur = con.execute(
                    "INSERT INTO predictions(timestamp,symbol,signal,pipeline_version,payload) VALUES(?,?,?,?,?)",
                    (result.timestamp.isoformat(), result.symbol, result.signal.value, result.pipeline_version, json.dumps(payload, sort_keys=True)),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("shadow prediction already exists; immutable record cannot be overwritten") from exc
            return int(cur.lastrowid)

    def attach_outcome(self, prediction_id: int, outcome: dict[str, object]) -> None:
        with self._connect() as con:
            try:
                con.execute("INSERT INTO outcomes(prediction_id,payload) VALUES(?,?)", (prediction_id, json.dumps(outcome, sort_keys=True)))
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("shadow outcome already attached or prediction does not exist") from exc

    def list_predictions(self) -> tuple[ShadowPrediction, ...]:
        with self._connect() as con:
            rows = con.execute("SELECT id,timestamp,symbol,signal,pipeline_version FROM predictions ORDER BY id").fetchall()
        return tuple(ShadowPrediction(int(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])) for r in rows)
