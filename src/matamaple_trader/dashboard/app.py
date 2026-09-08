from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .viewmodel import DashboardSnapshot

RUNTIME_JOURNAL = Path("data/fbs/runtime/demo_runtime.jsonl")


CSS = """
<style>
.stApp { background: #07111f; color: #e8eef7; }
.block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background: #0b1727; border-right: 1px solid #1b2b40; }
[data-testid="stMetric"] { background: #0d1b2c; border: 1px solid #20334b; padding: 14px 16px; border-radius: 18px; }
.mm-hero { background: linear-gradient(135deg,#0f2239 0%,#102d49 55%,#12395a 100%); border:1px solid #244765; border-radius:24px; padding:24px 28px; margin-bottom:18px; box-shadow:0 12px 30px rgba(0,0,0,.18); }
.mm-title { font-size:2rem; font-weight:800; letter-spacing:-.03em; margin:0; }
.mm-sub { color:#9fb0c6; margin-top:6px; font-size:.96rem; }
.mm-badge { display:inline-block; padding:6px 10px; border-radius:999px; background:#12314b; border:1px solid #285271; color:#b9daf1; font-size:.78rem; margin-right:6px; }
.mm-card { background:#0d1b2c; border:1px solid #20334b; border-radius:18px; padding:16px 18px; min-height:122px; }
.mm-card h4 { margin:0 0 8px 0; font-size:.9rem; color:#91a6bd; font-weight:600; }
.mm-big { font-size:1.55rem; font-weight:800; margin:0; }
.mm-good { color:#63e6a5; }
.mm-warn { color:#ffd166; }
.mm-bad { color:#ff7b7b; }
.mm-muted { color:#8fa3b8; font-size:.84rem; }
.mm-section { font-size:1.1rem; font-weight:750; margin:12px 0 10px 0; }
div[data-testid="stDataFrame"] { border:1px solid #20334b; border-radius:16px; overflow:hidden; }
.stButton>button { border-radius:12px; min-height:42px; font-weight:700; border:1px solid #2f5575; }
</style>
"""


def _row_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "__dataclass_fields__"):
        return asdict(row)
    return {
        "symbol": row.symbol,
        "signal": row.signal,
        "probability": row.probability,
        "regime": row.regime,
        "grade": row.grade,
        "risk_reward": row.risk_reward,
        "spread_state": row.spread_state,
        "operational_state": row.operational_state,
        "pipeline_version": row.pipeline_version,
        "warning": row.warning,
    }


def _load_runtime_records(path: Path = RUNTIME_JOURNAL, limit: int = 100) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            if not line.strip():
                continue
            records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _latest_runtime_summary(records: list[dict[str, Any]]) -> dict[str, str]:
    if not records:
        return {
            "signal": "WAIT",
            "symbol": "—",
            "ai": "Waiting",
            "risk": "Protected",
            "execution": "DRY RUN",
            "reason": "No runtime records yet",
        }
    latest = records[-1]
    quant = latest.get("quant", {})
    decision = latest.get("decision", {})
    ai_review = decision.get("ai_review") or {}
    return {
        "signal": str(decision.get("final_signal") or quant.get("signal") or "WAIT"),
        "symbol": str(quant.get("symbol") or "—"),
        "ai": str(ai_review.get("decision") or "Not called"),
        "risk": "PASS" if decision.get("risk", {}).get("allowed") else "GUARDED",
        "execution": "SUBMITTED" if decision.get("submitted") else "SAFE",
        "reason": str(decision.get("reason") or "No decision reason"),
    }


def _signal_class(signal: str) -> str:
    normalized = signal.upper()
    if normalized in {"BUY", "SELL"}:
        return "mm-good"
    if normalized == "WAIT":
        return "mm-warn"
    return "mm-bad"


def run_dashboard(snapshot: DashboardSnapshot | None = None) -> None:
    """Clean Streamlit control surface for FBS demo operation.

    The dashboard is intentionally an observer/control surface. It does not bypass
    the deterministic runtime, risk engine, execution guard, or live-trading lock.
    """
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("dashboard requires optional 'ui' dependencies") from exc

    st.set_page_config(
        page_title="MATAMAPLE TRADER",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    records = _load_runtime_records()
    latest = _latest_runtime_summary(records)

    with st.sidebar:
        st.markdown("## MATAMAPLE")
        st.caption("FBS Demo Trading Console")
        st.divider()
        page = st.radio("เมนู", ["หน้าหลัก", "สัญญาณ", "ระบบและความเสี่ยง"], label_visibility="collapsed")
        st.divider()
        st.caption("Execution mode")
        st.markdown("**DRY RUN / DEMO only**")
        st.caption("LIVE trading ถูกล็อกโดยระบบ")
        if st.button("↻ รีเฟรชข้อมูล", use_container_width=True):
            st.rerun()

    st.markdown(
        """
        <div class="mm-hero">
          <div class="mm-title">MATAMAPLE TRADER</div>
          <div class="mm-sub">หน้าควบคุมหลักสำหรับ FBS MT5 • Quant + Qwen AI + Risk Engine</div>
          <div style="margin-top:14px">
            <span class="mm-badge">FBS MT5</span>
            <span class="mm-badge">Qwen Local AI</span>
            <span class="mm-badge">Fail-Closed</span>
            <span class="mm-badge">LIVE Locked</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "หน้าหลัก":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FBS", "พร้อมเชื่อมต่อ", "MT5")
        c2.metric("AI Reviewer", latest["ai"], "Qwen")
        c3.metric("Risk Engine", latest["risk"], "Fail-closed")
        c4.metric("Execution", latest["execution"], "LIVE locked")

        st.markdown('<div class="mm-section">สัญญาณล่าสุด</div>', unsafe_allow_html=True)
        left, right = st.columns([1.45, 1])
        with left:
            sig_class = _signal_class(latest["signal"])
            st.markdown(
                f"""
                <div class="mm-card">
                    <h4>{latest['symbol']} • Current Decision</h4>
                    <p class="mm-big {sig_class}">{latest['signal']}</p>
                    <div class="mm-muted">{latest['reason']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                """
                <div class="mm-card">
                    <h4>Safety Status</h4>
                    <p class="mm-big mm-good">PROTECTED</p>
                    <div class="mm-muted">Risk → Guard → Preflight → order_check → Demo Sender</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="mm-section">ภาพรวมระบบ</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        active = snapshot.active if snapshot else 0
        degraded = snapshot.degraded if snapshot else 0
        paused = snapshot.paused if snapshot else 0
        a.metric("Active", active)
        b.metric("Degraded", degraded)
        c.metric("Paused", paused)

        if not records and not (snapshot and snapshot.rows):
            st.info("ยังไม่มีข้อมูล runtime บนเครื่องนี้ เริ่มจาก FBS readiness / DRY RUN แล้วข้อมูลจะขึ้นในหน้านี้อัตโนมัติ")

    elif page == "สัญญาณ":
        st.markdown('<div class="mm-section">Recent Signals</div>', unsafe_allow_html=True)
        rows = [_row_dict(row) for row in snapshot.rows] if snapshot else []
        if rows:
            frame = pd.DataFrame(rows)
            preferred = ["symbol", "signal", "probability", "grade", "regime", "risk_reward", "spread_state", "warning"]
            st.dataframe(frame[[c for c in preferred if c in frame.columns]], use_container_width=True, hide_index=True)
        elif records:
            runtime_rows = []
            for item in reversed(records):
                q = item.get("quant", {})
                d = item.get("decision", {})
                runtime_rows.append({
                    "time": item.get("timestamp"),
                    "symbol": q.get("symbol"),
                    "quant": q.get("signal"),
                    "final": d.get("final_signal"),
                    "stage": d.get("stage"),
                    "reason": d.get("reason"),
                    "submitted": d.get("submitted", False),
                })
            st.dataframe(pd.DataFrame(runtime_rows), use_container_width=True, hide_index=True)
        else:
            st.info("ยังไม่มี signal history")

    else:
        st.markdown('<div class="mm-section">ระบบและความเสี่ยง</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="mm-card">
                    <h4>AI Safety</h4>
                    <p class="mm-big mm-good">Circuit Breaker ON</p>
                    <div class="mm-muted">AI มีสิทธิ์ยืนยัน/ลดระดับสัญญาณเท่านั้น ไม่สามารถสร้างทิศทางใหม่หรือเพิ่มความเสี่ยง</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="mm-card">
                    <h4>Execution Safety</h4>
                    <p class="mm-big mm-good">LIVE DISABLED</p>
                    <div class="mm-muted">Demo order ต้องผ่าน Risk, Execution Guard, Margin/P&L Preflight, order_check และ Kill Switch</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.warning("หน้าจอนี้ไม่สามารถปลด LIVE trading ได้ การปลด LIVE ต้องเป็น milestone แยกหลัง Demo evidence ผ่านเกณฑ์เท่านั้น")


def main() -> None:
    run_dashboard(None)


if __name__ == "__main__":
    main()
