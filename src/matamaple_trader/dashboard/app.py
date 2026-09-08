from __future__ import annotations

from .viewmodel import DashboardSnapshot


def run_dashboard(snapshot: DashboardSnapshot) -> None:
    """Optional Streamlit renderer. Import is lazy so core/CI does not require Streamlit."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("dashboard requires optional 'ui' dependencies") from exc

    st.set_page_config(page_title="MATAMAPLE TRADER", layout="wide")
    st.title("MATAMAPLE TRADER — Signal Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("ACTIVE", snapshot.active)
    c2.metric("DEGRADED", snapshot.degraded)
    c3.metric("PAUSED", snapshot.paused)
    st.dataframe([row.__dict__ if hasattr(row, "__dict__") else {
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
    } for row in snapshot.rows], use_container_width=True)
