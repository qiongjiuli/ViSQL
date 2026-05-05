"""Streamlit UI for ViSQL v2 — talks to the Flask backend over HTTP.

Layout:
  - Sidebar: dataset picker, question, optional reference image upload, Run button.
  - Main: tabs — Report / Chart / Data / SQL / Plan / Branch output / Style spec.

The Streamlit app is stateless. The pipeline (which holds GBs of model weights)
lives in a Flask backend that runs in the notebook process — see backend.py.
"""
from __future__ import annotations
import io
import os
import json
import time
import base64
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = os.environ.get("VISQL_BACKEND", "http://127.0.0.1:8765")

st.set_page_config(page_title="ViSQL v2", page_icon="📊", layout="wide")

# ── Header ───────────────────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.title("ViSQL v2")
    st.caption("Vision-Augmented Autonomous Data Scientist · EECS 6895")
with right:
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
        st.success(f"Backend: {h.get('status', 'ok')}")
        st.caption(f"BQ project: `{h.get('project', '?')}`")
    except Exception as e:
        st.error(f"Backend offline ({e}).")

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Inputs")
    db_id = st.selectbox("Dataset",
                         options=["thelook", "thelook_with_experiment", "sec", "ga_sample"],
                         index=0)
    question = st.text_area(
        "Question",
        value="Show me the top 10 product categories by revenue.",
        height=110,
    )
    ref_img = st.file_uploader("Reference chart (optional)", type=["png", "jpg", "jpeg"])
    run_btn = st.button("▶ Run", type="primary", use_container_width=True)

# ── Run ──────────────────────────────────────────────────────────
if run_btn:
    payload = {"question": question, "db_id": db_id}
    if ref_img is not None:
        payload["reference_image_b64"] = base64.b64encode(ref_img.read()).decode("ascii")

    with st.spinner("Running pipeline..."):
        t0 = time.time()
        try:
            r = requests.post(f"{BACKEND_URL}/run", json=payload, timeout=240)
            r.raise_for_status()
            result = r.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()
        elapsed = time.time() - t0

    # ── Status banner ──────────────────────────────────────────
    if result.get("error"):
        st.error(f"Pipeline error: {result['error']}")
    else:
        st.success(f"✓ Done in {elapsed:.1f}s")

    # ── Top KPIs row ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    route = result.get("route", {}) or {}
    label = route.get("label", "?")
    if route.get("gated_from"):
        c1.metric("Route", label, delta=f"gated from {route['gated_from']}",
                  delta_color="off")
    else:
        c1.metric("Route", label, delta=f"conf {route.get('confidence', 0):.2f}",
                  delta_color="off")
    c2.metric("SQL retries", result.get("n_retries", 0))
    c3.metric("Rows returned", result.get("n_rows", 0))
    c4.metric("Status", "✗" if result.get("error") else "✓")

    # If gated, show the rationale prominently
    if route.get("gated_from"):
        st.info(f"**Gate fired**: {route.get('rationale', '')}")

    # ── Tabs ──────────────────────────────────────────────────
    tab_report, tab_chart, tab_data, tab_sql, tab_plan, tab_branch, tab_style = st.tabs(
        ["📝 Report", "📊 Chart", "📋 Data", "💾 SQL", "🧠 Plan", "⚙ Branch output", "🎨 Style spec"]
    )

    with tab_report:
        report = result.get("report") or "(no report generated)"
        # Belt-and-suspenders: escape $ so Streamlit doesn't render it as LaTeX
        st.markdown(report.replace("$", r"\$"))

    with tab_chart:
        chart_b64s = result.get("chart_pngs_b64") or []
        if chart_b64s:
            for b in chart_b64s:
                st.image(base64.b64decode(b), use_column_width=True)
        else:
            st.info("No chart was rendered.")

    with tab_data:
        rows = result.get("rows") or []
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=400)
            st.caption(f"{len(df)} rows × {len(df.columns)} cols")
        else:
            st.info("No data returned.")

    with tab_sql:
        st.code(result.get("final_sql") or "(no SQL)", language="sql")
        attempts = result.get("sql_attempts") or []
        if attempts:
            st.subheader("Self-correction trace")
            for a in attempts:
                with st.expander(f"Attempt {a['attempt']} — {a.get('phase', '?')}"):
                    st.code(a.get("sql", ""), language="sql")
                    if a.get("error"):
                        st.error(a["error"])
                    elif a.get("n_rows") is not None:
                        st.success(f"Executed: {a['n_rows']} rows")

    with tab_plan:
        plan = result.get("plan") or {}
        if plan.get("reasoning"):
            st.markdown("**Reasoning trace** (CoT planner)")
            st.code(plan["reasoning"], language="markdown")
        st.markdown("**Structured plan**")
        st.json({k: v for k, v in plan.items() if k != "reasoning"})

    with tab_branch:
        branch = result.get("branch_output") or {}
        st.json(branch)

    with tab_style:
        spec = result.get("style_spec") or {}
        if spec:
            st.json(spec)
            cols = st.columns(len(spec.get("palette", [])) or 1)
            for col, hex_ in zip(cols, spec.get("palette", [])):
                col.color_picker(hex_, hex_, label_visibility="collapsed")
        else:
            st.info("No reference image uploaded.")

else:
    st.info(
        "**Welcome.** Pick a dataset, type a question, "
        "optionally upload a chart screenshot for style imitation, then click Run.\n\n"
        "**Demo queries to try:**\n"
        "- `Show me the top 10 product categories by revenue.` (theLook · single_chart)\n"
        "- `For the checkout_redesign_2024 A/B test, did treatment lift conversion vs control?` "
        "(thelook_with_experiment · ab_test)\n"
        "- `Is conversion significantly different between mobile and desktop users?` "
        "(thelook · should be **gated** to single_chart)\n"
    )
