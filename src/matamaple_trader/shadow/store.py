from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
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
            con.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, symbol TEXT NOT NULL, signal TEXT NOT NULL, pipeline_version TEXT NOT NULL, payload TEXT NOT NULL, payload_sha256 TEXT, UNIQUE(timestamp, symbol, pipeline_version))")
            con.execute("CREATE TABLE IF NOT EXISTS outcomes (prediction_id INTEGER PRIMARY KEY, payload TEXT NOT NULL, payload_sha256 TEXT, FOREIGN KEY(prediction_id) REFERENCES predictions(id))")
            cols={r[1] for r in con.execute('PRAGMA table_info(predictions)').fetchall()}
            if 'payload_sha256' not in cols: con.execute('ALTER TABLE predictions ADD COLUMN payload_sha256 TEXT')
            ocols={r[1] for r in con.execute('PRAGMA table_info(outcomes)').fetchall()}
            if 'payload_sha256' not in ocols: con.execute('ALTER TABLE outcomes ADD COLUMN payload_sha256 TEXT')

    def _connect(self) -> sqlite3.Connection:
        con=sqlite3.connect(self.path)
        con.execute('PRAGMA foreign_keys=ON')
        return con

    @staticmethod
    def _encode(payload:dict[str,object])->tuple[str,str]:
        text=json.dumps(payload,sort_keys=True,separators=(',',':'))
        return text,hashlib.sha256(text.encode()).hexdigest()

    def append_prediction(self, result: SignalResult, *, observation: dict[str, object] | None = None) -> int:
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
            "observation": observation or {},
        }
        text,digest=self._encode(payload)
        with self._connect() as con:
            try:
                cur = con.execute(
                    "INSERT INTO predictions(timestamp,symbol,signal,pipeline_version,payload,payload_sha256) VALUES(?,?,?,?,?,?)",
                    (result.timestamp.isoformat(), result.symbol, result.signal.value, result.pipeline_version, text, digest),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("shadow prediction already exists; immutable record cannot be overwritten") from exc
            return int(cur.lastrowid)

    def attach_outcome(self, prediction_id: int, outcome: dict[str, object]) -> None:
        text,digest=self._encode(outcome)
        with self._connect() as con:
            try:
                con.execute("INSERT INTO outcomes(prediction_id,payload,payload_sha256) VALUES(?,?,?)", (prediction_id,text,digest))
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("shadow outcome already attached or prediction does not exist") from exc

    def pending_outcomes(self) -> tuple[dict[str, object], ...]:
        with self._connect() as con:
            rows=con.execute("SELECT p.id,p.timestamp,p.symbol,p.signal,p.pipeline_version,p.payload,p.payload_sha256 FROM predictions p LEFT JOIN outcomes o ON o.prediction_id=p.id WHERE o.prediction_id IS NULL ORDER BY p.id").fetchall()
        result=[]
        for row in rows:
            payload=str(row[5]); digest=str(row[6] or '')
            if digest and hashlib.sha256(payload.encode()).hexdigest()!=digest:
                raise RuntimeError('shadow prediction integrity check failed')
            result.append({'prediction_id':int(row[0]),'timestamp':str(row[1]),'symbol':str(row[2]),'signal':str(row[3]),'pipeline_version':str(row[4]),'payload':json.loads(payload)})
        return tuple(result)

    def list_predictions(self) -> tuple[ShadowPrediction, ...]:
        with self._connect() as con:
            rows = con.execute("SELECT id,timestamp,symbol,signal,pipeline_version FROM predictions ORDER BY id").fetchall()
        return tuple(ShadowPrediction(int(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])) for r in rows)

    def stats(self) -> dict[str,float|int]:
        with self._connect() as con:
            predictions=int(con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0])
            outcomes=int(con.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0])
        coverage=float(outcomes/predictions) if predictions else 0.0
        return {'predictions':predictions,'outcomes':outcomes,'outcome_coverage':coverage}
