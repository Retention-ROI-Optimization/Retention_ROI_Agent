import hashlib
import os
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.services.churn_service import get_churn_status
from dashboard.services.cohort_service import (
    get_cohort_curve,
    get_cohort_display_table,
    get_cohort_pivot,
    get_cohort_summary,
)
from dashboard.services.data_loader import load_dashboard_bundle
from dashboard.services.llm_service import (
    DEFAULT_MODEL_NAME,
    answer_dashboard_question,
    build_payload_json,
    dataframe_snapshot,
    generate_dashboard_summary,
    get_llm_status,
    numeric_summary,
    series_distribution,
)
from dashboard.services.optimize_service import get_budget_result
from dashboard.services.uplift_service import (
    get_retention_targets,
    get_top_high_value_customers,
)
from dashboard.utils.formatters import money, pct


st.set_page_config(
    page_title="Retention ROI Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_app_data():
    return load_dashboard_bundle()


def _payload_hash(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def get_session_cached_summary(
    view_title: str,
    payload_json: str,
    api_key: str,
    model_name: str,
) -> str:
    cache_key = f"summary::{_payload_hash(view_title, payload_json, model_name)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = generate_dashboard_summary(
            view_title=view_title,
            payload_json=payload_json,
            user_api_key=api_key,
            model_name=model_name,
        )
    return st.session_state[cache_key]


def get_session_cached_answer(
    view_title: str,
    payload_json: str,
    question: str,
    api_key: str,
    model_name: str,
) -> str:
    cache_key = f"qa::{_payload_hash(view_title, payload_json, question, model_name)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = answer_dashboard_question(
            view_title=view_title,
            payload_json=payload_json,
            question=question,
            user_api_key=api_key,
            model_name=model_name,
        )
    return st.session_state[cache_key]


def render_llm_panel(
    view_key: str,
    view_title: str,
    payload: Dict,
    api_key: Optional[str],
    model_name: str,
):
    st.divider()
    st.subheader("LLM 결과 요약")
    st.caption("현재 화면의 지표·표·그래프에서 추린 요약 컨텍스트만 바탕으로 응답합니다.")

    ready, status_message = get_llm_status(api_key)
    payload_json = build_payload_json(payload)

    if not ready:
        st.info(status_message)
        return

    with st.spinner("AI가 현재 화면의 결과를 요약하는 중입니다..."):
        try:
            summary = get_session_cached_summary(
                view_title=view_title,
                payload_json=payload_json,
                api_key=api_key or "",
                model_name=model_name,
            )
        except Exception as exc:
            st.error(f"AI 요약 생성 중 오류가 발생했습니다: {exc}")
            return

    st.markdown(summary)

    st.markdown("### 결과 지표에 대해 질문하기")
    st.caption(
        "예: 왜 특정 세그먼트가 많이 선정됐는지 설명해줘 / 현재 예산에서 ROI가 가장 높은 세그먼트는?"
    )

    history_key = f"llm_history_{view_key}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    question = st.text_area(
        "질문 입력",
        key=f"llm_question_{view_key}",
        height=100,
        placeholder="현재 화면의 지표에 대해 질문을 입력하세요.",
    )

    q_col1, q_col2 = st.columns([1, 1])
    ask_clicked = q_col1.button("AI에게 질문하기", key=f"ask_{view_key}")
    clear_clicked = q_col2.button("대화 지우기", key=f"clear_{view_key}")

    if clear_clicked:
        st.session_state[history_key] = []

    if ask_clicked:
        user_question = question.strip()
        if not user_question:
            st.warning("질문을 먼저 입력하세요.")
        else:
            with st.spinner("AI가 질문에 답변하는 중입니다..."):
                try:
                    answer = get_session_cached_answer(
                        view_title=view_title,
                        payload_json=payload_json,
                        question=user_question,
                        api_key=api_key or "",
                        model_name=model_name,
                    )
                except Exception as exc:
                    st.error(f"AI 답변 생성 중 오류가 발생했습니다: {exc}")
                else:
                    st.session_state[history_key].append(
                        {"question": user_question, "answer": answer}
                    )

    if st.session_state[history_key]:
        for idx, item in enumerate(reversed(st.session_state[history_key]), start=1):
            with st.container():
                st.markdown(f"**Q{idx}.** {item['question']}")
                st.markdown(item["answer"])
                st.divider()


bundle = load_app_data()
customers = bundle.customer_summary
cohort_df = bundle.cohort_retention

st.title("AI 기반 고객 이탈 예측 및 리텐션 ROI 최적화 대시보드")
if bundle.used_mock:
    st.warning("실제 data/raw 산출물을 찾지 못해 mock data로 실행 중입니다.")

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
        ],
    )

    threshold = 0.50
    budget = 5_000_000
    top_n = 20
    target_cap = 1000

    if view in {"1. 이탈현황", "4. 예산 배분 결과", "5. 예상 최적화 ROI", "6. 리텐션 대상 고객 목록"}:
        threshold = st.slider(
            "이탈 Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.01,
            help="이 값 이상인 고객을 이탈 위험군으로 간주합니다.",
        )

    if view in {"3. Uplift + CLV 상위 고객", "6. 리텐션 대상 고객 목록"}:
        top_n = st.slider(
            "표시 고객 수",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
        )

    if view in {"4. 예산 배분 결과", "5. 예상 최적화 ROI"}:
        budget = st.number_input(
            "총 마케팅 예산",
            min_value=100000,
            max_value=100000000,
            value=5000000,
            step=100000,
        )
        target_cap = st.slider(
            "최대 타겟 고객 수",
            min_value=50,
            max_value=3000,
            value=1000,
            step=50,
            help="예산이 충분하더라도 이 수를 넘겨 타겟팅하지 않습니다.",
        )

    st.divider()
    st.subheader("LLM 설정")
    st.caption(
        "권장: API 키는 코드에 쓰지 말고 환경변수 `OPENAI_API_KEY` 또는 Streamlit secrets로 관리하세요."
    )

    llm_enabled = st.toggle("LLM 요약/질문 기능 사용", value=True)
    llm_api_key = st.text_input(
        "OpenAI API Key (선택)",
        type="password",
        help="비워두면 OPENAI_API_KEY 환경변수를 사용합니다.",
    )
    llm_model = st.text_input("LLM 모델명", value=DEFAULT_MODEL_NAME)

    env_key_configured = bool(os.getenv("OPENAI_API_KEY"))
    if env_key_configured and not llm_api_key:
        st.caption("현재 OPENAI_API_KEY 환경변수를 사용하도록 설정되어 있습니다.")


churn_summary, risk_customers = get_churn_status(customers, threshold)
cohort_curve = get_cohort_curve(cohort_df)
top_customers = get_top_high_value_customers(customers, top_n=top_n)
selected_customers, optimize_summary, segment_allocation = get_budget_result(
    customers,
    budget=budget,
    threshold=threshold,
    max_customers=target_cap,
)
retention_targets = get_retention_targets(customers, threshold, top_n=top_n)


c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 고객 수", f"{churn_summary['total_customers']:,}")
c2.metric("이탈 위험 고객 수", f"{churn_summary['at_risk_customers']:,}")
c3.metric("위험 고객 비율", pct(churn_summary["risk_rate"]))
c4.metric("평균 이탈 확률", pct(churn_summary["avg_churn_prob"]))

st.divider()

llm_view_title = view
llm_payload: Dict = {}
llm_api_key_value = llm_api_key.strip() if llm_api_key else None

if view == "1. 이탈현황":
    st.subheader("이탈현황")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        hist_fig = px.histogram(
            customers,
            x="churn_probability",
            nbins=30,
            title="고객별 이탈 확률 분포",
        )
        hist_fig.add_vline(
            x=threshold,
            line_dash="dash",
            annotation_text=f"Threshold={threshold:.2f}",
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
            title="페르소나별 이탈 위험 고객 수",
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

    llm_payload = {
        "threshold": threshold,
        "kpis": churn_summary,
        "all_customer_numeric_summary": numeric_summary(
            customers, ["churn_probability", "uplift_score", "clv", "expected_roi"]
        ),
        "persona_risk_counts": persona_risk.to_dict(orient="records"),
        "top_risk_customers": dataframe_snapshot(
            risk_customers,
            columns=[
                "customer_id",
                "persona",
                "churn_probability",
                "clv",
                "uplift_score",
                "uplift_segment",
            ],
            max_rows=min(top_n, 12),
        ),
    }

elif view == "2. 코호트 리텐션 곡선":
    st.subheader("코호트 리텐션 분석")

    cohort_summary = get_cohort_summary(cohort_df)
    display_table = get_cohort_display_table(cohort_df)
    heatmap_df = get_cohort_pivot(cohort_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("코호트 수", f"{cohort_summary['cohort_count']:,}")
    avg_size = cohort_summary["avg_cohort_size"]
    m2.metric("평균 코호트 크기", "-" if pd.isna(avg_size) else f"{avg_size:,.0f}")
    month1_ret = cohort_summary["month1_avg_retention"]
    m3.metric("평균 1개월차 리텐션", "-" if pd.isna(month1_ret) else f"{month1_ret:.2%}")
    last_avg_ret = cohort_summary["last_observed_avg_retention"]
    m4.metric("마지막 관측 리텐션 평균", "-" if pd.isna(last_avg_ret) else f"{last_avg_ret:.2%}")

    st.caption(
        "period 0은 코호트 정의상 100%로 고정하고, 아직 관측할 수 없는 미래 period는 0이 아니라 공란으로 둡니다. "
        "그래야 최근 코호트가 오른쪽 검열(right censoring) 때문에 과소평가되지 않습니다."
    )

    if cohort_curve.empty:
        st.warning("표시할 코호트 데이터가 없습니다.")
        last_period_df = cohort_curve.copy()
    else:
        line_fig = px.line(
            cohort_curve,
            x="period",
            y="retention_rate",
            color="cohort_month",
            markers=True,
            title="가입 코호트별 리텐션 곡선",
        )
        line_fig.update_layout(xaxis_title="경과 기간(개월)", yaxis_title="Retention Rate")
        st.plotly_chart(line_fig, use_container_width=True)

        if not heatmap_df.empty:
            heatmap_fig = px.imshow(
                heatmap_df,
                text_auto=".0%",
                aspect="auto",
                labels={"x": "경과 기간(개월)", "y": "코호트", "color": "Retention"},
                title="코호트 리텐션 히트맵",
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

        st.markdown("### 코호트 리텐션 테이블")
        st.dataframe(display_table, use_container_width=True, hide_index=True)

        last_period_df = (
            cohort_curve.sort_values(["cohort_month", "period"])
            .groupby("cohort_month", as_index=False)
            .tail(1)
            .sort_values("retention_rate", ascending=False)
            .reset_index(drop=True)
        )

        best_last = cohort_summary.get("best_last_cohort")
        worst_last = cohort_summary.get("worst_last_cohort")
        if best_last and worst_last:
            st.markdown(
                f"**최종 관측 기준 최고 코호트:** {best_last['cohort_month']} (period {best_last['period']}, {best_last['retention_rate']:.2%})"
                f"**최종 관측 기준 최저 코호트:** {worst_last['cohort_month']} (period {worst_last['period']}, {worst_last['retention_rate']:.2%})"
            )

    llm_payload = {
        "cohort_summary": cohort_summary,
        "retention_curve_summary": numeric_summary(cohort_curve, ["retention_rate"]),
        "cohort_retention_records": cohort_curve.round(4).to_dict(orient="records"),
        "last_observed_retention": last_period_df.round(4).to_dict(orient="records"),
    }

elif view == "3. Uplift + CLV 상위 고객":
    st.subheader("Uplift Score + CLV 상위 고가치 고객 목록")

    plot_df = top_customers.copy()
    plot_df["customer_label"] = plot_df["customer_id"].astype(str)
    plot_df["bubble_size"] = plot_df["value_score"].clip(lower=0.01)

    scatter_fig = px.scatter(
        plot_df,
        x="uplift_score",
        y="clv",
        size="bubble_size",
        color="uplift_segment",
        hover_data=[
            "customer_id",
            "persona",
            "churn_probability",
            "expected_incremental_profit",
            "value_score",
        ],
        title="상위 고객의 Uplift-CLV 분포",
        labels={"bubble_size": "value_score"},
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.caption(
        "버블 크기는 expected_incremental_profit 대신 value_score(CLV × uplift_score)를 사용합니다. "
        "일부 고객의 expected_incremental_profit이 음수가 될 수 있어, 상위 고객 수가 커질 때 Plotly size 오류를 막습니다."
    )

    display_df = plot_df[
        [
            "customer_id",
            "persona",
            "churn_probability",
            "uplift_score",
            "clv",
            "value_score",
            "expected_incremental_profit",
            "uplift_segment",
        ]
    ].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
    display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
    display_df["clv"] = display_df["clv"].map(money)
    display_df["value_score"] = display_df["value_score"].map(money)
    display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    llm_payload = {
        "top_n": top_n,
        "segment_distribution": series_distribution(plot_df, "uplift_segment"),
        "numeric_summary": numeric_summary(
            plot_df,
            ["uplift_score", "clv", "churn_probability", "expected_incremental_profit"],
        ),
        "top_customers": dataframe_snapshot(
            plot_df,
            columns=[
                "customer_id",
                "persona",
                "churn_probability",
                "uplift_score",
                "clv",
                "expected_incremental_profit",
                "uplift_segment",
            ],
            max_rows=min(top_n, 15),
        ),
    }

elif view == "4. 예산 배분 결과":
    st.subheader("예산 배분 결과")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 예산", money(optimize_summary["budget"]))
    m2.metric("집행 예산", money(optimize_summary["spent"]))
    m3.metric("잔여 예산", money(optimize_summary["remaining"]))
    m4.metric("타겟 고객 수", f"{optimize_summary['num_targeted']:,}")

    st.caption(
        f"이탈 확률 {threshold:.2f} 이상, uplift_score > 0, expected_incremental_profit > 0 인 고객을 후보군으로 두고 "
        f"ROI 우선순위로 선정했습니다. 최대 타겟 고객 수는 {optimize_summary['max_customers_cap']:,}명입니다."
    )

    candidate_by_segment = pd.DataFrame(
        {
            "uplift_segment": list(optimize_summary.get("candidate_segment_counts", {}).keys()),
            "candidate_customer_count": list(optimize_summary.get("candidate_segment_counts", {}).values()),
        }
    )

    if not candidate_by_segment.empty:
        cand_fig = px.bar(
            candidate_by_segment,
            x="uplift_segment",
            y="candidate_customer_count",
            text="candidate_customer_count",
            title="세그먼트별 예산 배분 후보 고객 수",
        )
        st.plotly_chart(cand_fig, use_container_width=True)

    if segment_allocation.empty or optimize_summary["num_targeted"] == 0:
        st.warning("현재 조건에서 예산 배분 대상 고객이 없습니다.")
    else:
        bar_fig = px.bar(
            segment_allocation,
            x="uplift_segment",
            y="allocated_budget",
            text="customer_count",
            title="세그먼트별 예산 배분",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        display_df = segment_allocation.copy()
        display_df["allocated_budget"] = display_df["allocated_budget"].map(money)
        display_df["expected_profit"] = display_df["expected_profit"].map(money)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    llm_payload = {
        "budget_summary": optimize_summary,
        "segment_allocation": segment_allocation.round(4).to_dict(orient="records"),
        "candidate_segment_counts": optimize_summary.get("candidate_segment_counts", {}),
        "selected_customer_overview": dataframe_snapshot(
            selected_customers,
            columns=[
                "customer_id",
                "persona",
                "uplift_segment",
                "coupon_cost",
                "expected_incremental_profit",
                "expected_roi",
            ],
            max_rows=min(12, len(selected_customers)),
        ),
        "selected_customer_numeric_summary": numeric_summary(
            selected_customers, ["coupon_cost", "expected_incremental_profit", "expected_roi"]
        ),
    }

elif view == "5. 예상 최적화 ROI":
    st.subheader("예상 최적화 ROI")

    m1, m2, m3 = st.columns(3)
    m1.metric("예상 증분 이익", money(optimize_summary["expected_incremental_profit"]))
    m2.metric("예상 ROI", pct(optimize_summary["overall_roi"]))
    m3.metric("선정 고객 수", f"{optimize_summary['num_targeted']:,}")

    top_roi = selected_customers.copy()
    if selected_customers.empty:
        st.warning("현재 조건에서 ROI 계산 대상이 없습니다.")
    else:
        roi_fig = px.histogram(
            selected_customers,
            x="expected_roi",
            nbins=25,
            title="선정 고객의 예상 ROI 분포",
        )
        st.plotly_chart(roi_fig, use_container_width=True)

        st.caption(
            f"후보 고객 {optimize_summary['candidate_customers']:,}명 중 예산 및 최대 타겟 고객 수 제약 아래 선정된 고객만 포함합니다."
        )

        top_roi = selected_customers.sort_values("expected_roi", ascending=False).head(min(20, len(selected_customers)))
        display_df = top_roi[
            [
                "customer_id",
                "persona",
                "uplift_score",
                "clv",
                "coupon_cost",
                "expected_incremental_profit",
                "expected_roi",
            ]
        ].copy()
        display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
        display_df["clv"] = display_df["clv"].map(money)
        display_df["coupon_cost"] = display_df["coupon_cost"].map(money)
        display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)
        display_df["expected_roi"] = display_df["expected_roi"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    llm_payload = {
        "optimize_summary": optimize_summary,
        "roi_numeric_summary": numeric_summary(
            selected_customers,
            ["expected_roi", "coupon_cost", "expected_incremental_profit"],
        ),
        "top_roi_customers": dataframe_snapshot(
            top_roi,
            columns=[
                "customer_id",
                "persona",
                "uplift_score",
                "clv",
                "coupon_cost",
                "expected_incremental_profit",
                "expected_roi",
            ],
            max_rows=min(top_n, 12),
        ),
    }

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
            title="우선순위 상위 리텐션 대상 고객",
        )
        st.plotly_chart(priority_fig, use_container_width=True)

        display_df = retention_targets[
            [
                "customer_id",
                "persona",
                "churn_probability",
                "uplift_score",
                "clv",
                "uplift_segment",
                "priority_score",
            ]
        ].copy()
        display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
        display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
        display_df["clv"] = display_df["clv"].map(money)
        display_df["priority_score"] = display_df["priority_score"].map(lambda x: f"{x:.3f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    llm_payload = {
        "threshold": threshold,
        "target_count": int(len(retention_targets)),
        "persona_distribution": series_distribution(retention_targets, "persona"),
        "segment_distribution": series_distribution(retention_targets, "uplift_segment"),
        "numeric_summary": numeric_summary(
            retention_targets, ["priority_score", "churn_probability", "uplift_score", "clv"]
        ),
        "retention_targets": dataframe_snapshot(
            retention_targets,
            columns=[
                "customer_id",
                "persona",
                "churn_probability",
                "uplift_score",
                "clv",
                "uplift_segment",
                "priority_score",
            ],
            max_rows=min(top_n, 15),
        ),
    }

if llm_enabled:
    render_llm_panel(
        view_key=view.split(".")[0],
        view_title=llm_view_title,
        payload=llm_payload,
        api_key=llm_api_key_value,
        model_name=llm_model.strip() or DEFAULT_MODEL_NAME,
    )
