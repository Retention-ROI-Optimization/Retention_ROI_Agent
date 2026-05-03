from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.services import universal_autoops_client as client
from src.autoops_universal.dashboard_contract import save_runtime_config

st.set_page_config(page_title="Universal AutoOps Onboarding", layout="wide")
st.title("Universal AutoOps Data Onboarding")
st.caption("Upload a customer-master or transaction CSV. The add-on validates whether the file is relevant to retention analysis, maps columns to a canonical schema, aggregates to customer level, retrains, and refreshes dashboard artifacts without editing legacy source code.")

with st.expander("How this handles different user schemas", expanded=False):
    st.markdown(
        """
        **Flow**: uploaded CSV → relevance validation → profiling → automatic schema mapping → grain detection → transaction-to-customer aggregation if needed → compatibility columns → model retraining → dashboard result refresh.

        You do not need to rename every uploaded column manually. If the file has no customer/member/name identifier and no purchase/activity/churn signal, it is rejected as an invalid file instead of being forced into the dashboard. If the automatic mapping is ambiguous, provide a small JSON override such as:

        ```json
        {"customer_id": "회원번호", "transaction_date": "주문일자", "amount": "결제금액", "label": "이탈여부"}
        ```
        """
    )

col_left, col_right = st.columns([1.05, 1.0], gap="large")

with col_left:
    st.subheader("1) Upload CSV")
    uploaded = st.file_uploader("Customer or transaction dataset CSV", type=["csv"], help="Customer-level or transaction-level CSV is accepted. Limit depends on Streamlit server configuration.")
    budget = st.number_input("Campaign budget", min_value=1, value=50_000_000, step=1_000_000)
    threshold = st.slider("Targeting churn threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    max_customers = st.number_input("Maximum target customers", min_value=1, value=1000, step=100)
    mapping_json = st.text_area(
        "Optional manual column mapping JSON",
        value="",
        placeholder='{"customer_id":"회원번호", "transaction_date":"주문일자", "amount":"결제금액"}',
        height=120,
    )
    if mapping_json.strip():
        try:
            json.loads(mapping_json)
            st.success("Manual mapping JSON is valid.")
        except Exception as exc:
            st.error(f"Manual mapping JSON is invalid: {exc}")

    if uploaded is not None:
        st.info(f"Selected file: {uploaded.name} ({uploaded.size / 1024 / 1024:.1f} MB)")
        if st.button("Upload → Map Schema → Retrain → Refresh Dashboard", type="primary"):
            try:
                with st.spinner("Uploading CSV and starting background job..."):
                    response = client.submit_onboarding_csv(
                        uploaded.getvalue(),
                        filename=uploaded.name,
                        budget=int(budget),
                        threshold=float(threshold),
                        max_customers=int(max_customers),
                        mapping_json=mapping_json if mapping_json.strip() else None,
                        timeout=120,
                    )
                save_runtime_config(result_dir=ROOT / "results", budget=int(budget), threshold=float(threshold), max_customers=int(max_customers))
                st.session_state["universal_autoops_job_id"] = response.get("job_id")
                st.success(f"Job started: {response.get('job_id')}")
            except Exception as exc:
                st.error(f"Universal AutoOps API call failed: {exc}")
                st.caption("Check that the API is running: RETENTION_API_PORT=8010 python scripts/run_universal_autoops_api.py")

with col_right:
    st.subheader("2) Current Status")
    try:
        st.json(client.health())
    except Exception as exc:
        st.warning(f"API health check failed: {exc}")

    job_id = st.session_state.get("universal_autoops_job_id")
    if job_id:
        st.markdown(f"**Current job:** `{job_id}`")
        if st.button("Refresh job status"):
            st.rerun()
        try:
            job = client.job(job_id)
            st.json(job)
            if job.get("status") == "running":
                st.info("Job is still running. Large transaction files can take several minutes.")
            elif job.get("status") == "ready":
                st.success("Job completed. Dashboard artifacts have been refreshed.")
            elif job.get("status") == "failed":
                st.error(job.get("error"))
                if job.get("traceback"):
                    st.code(job.get("traceback"))
        except Exception as exc:
            st.error(f"Job status check failed: {exc}")

    st.markdown("### Pipeline Status")
    try:
        st.json(client.status())
    except Exception as exc:
        st.warning(f"Status call failed: {exc}")

st.divider()
st.subheader("3) Generated Mapping and Diagnostics")
try:
    artifacts = client.artifacts()
    diagnostics = artifacts.get("diagnostics", {})
    mapping = artifacts.get("schema_mapping", {})
    c1, c2, c3, c4 = st.columns(4)
    profile = diagnostics.get("profile", {}) if isinstance(diagnostics, dict) else {}
    grain = diagnostics.get("grain_detection", {}) if isinstance(diagnostics, dict) else {}
    c1.metric("Input rows", f"{profile.get('rows', 0):,}" if profile else "-")
    c2.metric("Canonical customers", f"{diagnostics.get('canonical_rows', 0):,}" if diagnostics else "-")
    c3.metric("Detected grain", grain.get("grain", "-"))
    c4.metric("Label source", diagnostics.get("label_source", "-"))

    table_path = ROOT / "results" / "schema_mapping_table.csv"
    if table_path.exists():
        st.markdown("#### Schema Mapping Table")
        st.dataframe(pd.read_csv(table_path), use_container_width=True)
    else:
        st.info("Mapping table has not been generated yet.")

    if mapping:
        with st.expander("Raw mapping report"):
            st.json(mapping)
except Exception as exc:
    st.warning(f"Artifact call failed: {exc}")

st.caption("Run the dashboard with: PYTHONPATH=$PWD RETENTION_UNIVERSAL_AUTOOPS_API_BASE_URL=http://localhost:8010 streamlit run dashboard/app.py")
