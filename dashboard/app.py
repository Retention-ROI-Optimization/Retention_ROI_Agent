import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.data.mock_data import generate_mock_customers, generate_mock_cohort_retention
from dashboard.services.churn_service import get_churn_status
from dashboard.services.cohort_service import get_cohort_curve
from dashboard.services.uplift_service import get_top_high_value_customers, get_retention_targets
from dashboard.services.optimize_service import get_budget_result
from dashboard.utils.formatters import money, pct


st.set_page_config(
    page_title="Retention ROI Mock Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_mock_data():
    customers = generate_mock_customers(n_customers=500, seed=42)
    cohort = generate_mock_cohort_retention(seed=42)
    return customers, cohort


customers, cohort_df = load_mock_data()

st.title("AI 기반 고객 이탈 예측 및 리텐션 ROI 최적화 대시보드")
st.caption("Mock data 기반 데모 대시보드")

with st.sidebar:
    st.header("제어 패널")

    view = st.radio(
        "조회 항목 선택",
        [
            "1. 이탈현황",
            "2. 코호트 리텐션 곡선",
            "3. Uplift + CLV 상위 고객",
            "4. 예산 배분 결과",
            "5. 예상 최적화 ROI",
            "6. 리텐션 대상 고객 목록",
        ]
    )

    threshold = st.slider(
        "이탈 Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01
    )

    budget = st.number_input(
        "총 마케팅 예산",
        min_value=100000,
        max_value=100000000,
        value=5000000,
        step=100000
    )

    top_n = st.slider(
        "상위 고객 수",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

# 공통 계산
churn_summary, risk_customers = get_churn_status(customers, threshold)
cohort_curve = get_cohort_curve(cohort_df)
top_customers = get_top_high_value_customers(customers, top_n=top_n)
selected_customers, optimize_summary, segment_allocation = get_budget_result(customers, budget)
retention_targets = get_retention_targets(customers, threshold, top_n=top_n)

# 상단 공통 KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 고객 수", f"{churn_summary['total_customers']:,}")
c2.metric("이탈 위험 고객 수", f"{churn_summary['at_risk_customers']:,}")
c3.metric("위험 고객 비율", pct(churn_summary["risk_rate"]))
c4.metric("평균 이탈 확률", pct(churn_summary["avg_churn_prob"]))

st.divider()

if view == "1. 이탈현황":
    st.subheader("이탈현황")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        hist_fig = px.histogram(
            customers,
            x="churn_probability",
            nbins=30,
            title="고객별 이탈 확률 분포"
        )
        hist_fig.add_vline(
            x=threshold,
            line_dash="dash",
            annotation_text=f"Threshold={threshold:.2f}"
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        persona_risk = (
            risk_customers.groupby("persona", as_index=False)
            .agg(at_risk_count=("customer_id", "count"))
            .sort_values("at_risk_count", ascending=False)
        )

        bar_fig = px.bar(
            persona_risk,
            x="persona",
            y="at_risk_count",
            title="페르소나별 이탈 위험 고객 수"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("### 이탈 위험 고객 목록")
    display_df = risk_customers[
        ["customer_id", "persona", "churn_probability", "clv", "uplift_score", "uplift_segment"]
    ].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
    display_df["clv"] = display_df["clv"].map(money)
    display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif view == "2. 코호트 리텐션 곡선":
    st.subheader("코호트 리텐션 곡선")

    line_fig = px.line(
        cohort_curve,
        x="period",
        y="retention_rate",
        color="cohort_month",
        markers=True,
        title="가입 코호트별 리텐션 곡선"
    )
    line_fig.update_layout(
        xaxis_title="경과 기간",
        yaxis_title="Retention Rate"
    )
    st.plotly_chart(line_fig, use_container_width=True)

    pivot_df = cohort_curve.pivot(
        index="cohort_month",
        columns="period",
        values="retention_rate"
    ).reset_index()

    formatted_pivot = pivot_df.copy()
    for col in formatted_pivot.columns[1:]:
        formatted_pivot[col] = formatted_pivot[col].map(lambda x: f"{x:.2%}")

    st.markdown("### 코호트 리텐션 테이블")
    st.dataframe(formatted_pivot, use_container_width=True, hide_index=True)

elif view == "3. Uplift + CLV 상위 고객":
    st.subheader("Uplift Score + CLV 상위 고가치 고객 목록")

    plot_df = top_customers.copy()
    plot_df["customer_label"] = plot_df["customer_id"].astype(str)

    scatter_fig = px.scatter(
        plot_df,
        x="uplift_score",
        y="clv",
        size="expected_incremental_profit",
        color="uplift_segment",
        hover_data=["customer_id", "persona", "churn_probability"],
        title="상위 고객의 Uplift-CLV 분포"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    display_df = plot_df[
        ["customer_id", "persona", "churn_probability", "uplift_score", "clv", "expected_incremental_profit", "uplift_segment"]
    ].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
    display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
    display_df["clv"] = display_df["clv"].map(money)
    display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif view == "4. 예산 배분 결과":
    st.subheader("예산 배분 결과")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 예산", money(optimize_summary["budget"]))
    m2.metric("집행 예산", money(optimize_summary["spent"]))
    m3.metric("잔여 예산", money(optimize_summary["remaining"]))
    m4.metric("타겟 고객 수", f"{optimize_summary['num_targeted']:,}")

    if segment_allocation.empty:
        st.warning("현재 조건에서 예산 배분 대상 고객이 없습니다.")
    else:
        bar_fig = px.bar(
            segment_allocation,
            x="uplift_segment",
            y="allocated_budget",
            text="customer_count",
            title="세그먼트별 예산 배분"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        display_df = segment_allocation.copy()
        display_df["allocated_budget"] = display_df["allocated_budget"].map(money)
        display_df["expected_profit"] = display_df["expected_profit"].map(money)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

elif view == "5. 예상 최적화 ROI":
    st.subheader("예상 최적화 ROI")

    m1, m2, m3 = st.columns(3)
    m1.metric("예상 증분 이익", money(optimize_summary["expected_incremental_profit"]))
    m2.metric("예상 ROI", pct(optimize_summary["overall_roi"]))
    m3.metric("선정 고객 수", f"{optimize_summary['num_targeted']:,}")

    if selected_customers.empty:
        st.warning("현재 조건에서 ROI 계산 대상이 없습니다.")
    else:
        roi_fig = px.histogram(
            selected_customers,
            x="expected_roi",
            nbins=25,
            title="선정 고객의 예상 ROI 분포"
        )
        st.plotly_chart(roi_fig, use_container_width=True)

        top_roi = selected_customers.sort_values("expected_roi", ascending=False).head(top_n)
        display_df = top_roi[
            ["customer_id", "persona", "uplift_score", "clv", "coupon_cost", "expected_incremental_profit", "expected_roi"]
        ].copy()

        display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
        display_df["clv"] = display_df["clv"].map(money)
        display_df["coupon_cost"] = display_df["coupon_cost"].map(money)
        display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)
        display_df["expected_roi"] = display_df["expected_roi"].map(lambda x: f"{x:.2%}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

elif view == "6. 리텐션 대상 고객 목록":
    st.subheader("리텐션 대상 고객 목록")

    st.markdown(
        """
        선정 기준 예시:
        - churn_probability >= threshold
        - uplift_score > 0.08
        - CLV가 중간값 이상
        - Sleeping Dogs 제외
        """
    )

    if retention_targets.empty:
        st.warning("현재 조건에서 리텐션 타겟 고객이 없습니다.")
    else:
        priority_fig = px.bar(
            retention_targets.head(15),
            x="customer_id",
            y="priority_score",
            hover_data=["churn_probability", "uplift_score", "clv"],
            title="우선순위 상위 리텐션 대상 고객"
        )
        st.plotly_chart(priority_fig, use_container_width=True)

        display_df = retention_targets[
            ["customer_id", "persona", "churn_probability", "uplift_score", "clv", "uplift_segment", "priority_score"]
        ].copy()
        display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
        display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
        display_df["clv"] = display_df["clv"].map(money)
        display_df["priority_score"] = display_df["priority_score"].map(lambda x: f"{x:.3f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)