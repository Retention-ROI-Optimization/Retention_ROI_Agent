import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from dashboard.services.api_client import (
    fetch_personalized_recommendations,
    fetch_saved_results_artifacts,
    fetch_training_artifacts,
)
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



def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --bg-grad-1: #0f172a;
            --bg-grad-2: #111827;
            --card-bg: rgba(255,255,255,0.88);
            --card-border: rgba(15, 23, 42, 0.08);
            --accent: #2563eb;
            --accent-2: #7c3aed;
            --text-main: #0f172a;
            --text-soft: #475569;
            --success-bg: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(16,185,129,0.10));
            --warn-bg: linear-gradient(135deg, rgba(245,158,11,0.16), rgba(251,191,36,0.10));
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.08), transparent 28%),
                radial-gradient(circle at top right, rgba(124,58,237,0.08), transparent 22%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            color: var(--text-main);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* 사이드바 텍스트는 전체적으로 밝게 */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] * {
            color: #e5eefc !important;
        }

        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div div {
            background-color: rgba(255,255,255,0.18);
        }

        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(135deg, rgba(37,99,235,0.24), rgba(124,58,237,0.24));
            color: white !important;
            font-weight: 600;
        }

        /* radio / toggle / slider 글자 고정 */
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stRadio label p,
        section[data-testid="stSidebar"] .stToggle label,
        section[data-testid="stSidebar"] .stToggle label p,
        section[data-testid="stSidebar"] .stCheckbox label,
        section[data-testid="stSidebar"] .stCheckbox label p,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stSlider span,
        section[data-testid="stSidebar"] .stSlider p,
        section[data-testid="stSidebar"] [role="radiogroup"] label,
        section[data-testid="stSidebar"] [role="radiogroup"] label p {
            color: #e5eefc !important;
            -webkit-text-fill-color: #e5eefc !important;
            opacity: 1 !important;
        }

        /* 사이드바 입력칸은 흰 배경 + 진한 글씨로 원복 */
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stTextArea textarea,
        section[data-testid="stSidebar"] input[type="password"],
        section[data-testid="stSidebar"] input[type="text"],
        section[data-testid="stSidebar"] [data-baseweb="input"] input,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] input,
        section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea {
            border-radius: 14px !important;
            background: #ffffff !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            caret-color: #111827 !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] .stTextInput input::placeholder,
        section[data-testid="stSidebar"] .stNumberInput input::placeholder,
        section[data-testid="stSidebar"] .stTextArea textarea::placeholder,
        section[data-testid="stSidebar"] input[type="password"]::placeholder,
        section[data-testid="stSidebar"] input[type="text"]::placeholder,
        section[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] input::placeholder,
        section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"] .stNumberInput button,
        section[data-testid="stSidebar"] [data-baseweb="input"] button,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] button {
            color: #111827 !important;
            background: #ffffff !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
        }

        section[data-testid="stSidebar"] .stNumberInput button svg,
        section[data-testid="stSidebar"] [data-baseweb="input"] button svg,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] button svg {
            fill: #111827 !important;
            color: #111827 !important;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            padding: 32px 32px 26px 32px;
            margin-bottom: 18px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(37,99,235,0.92) 60%, rgba(124,58,237,0.88));
            box-shadow: 0 24px 60px rgba(15,23,42,0.22);
            color: white;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            inset: auto -70px -90px auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(255,255,255,0.22), transparent 65%);
            pointer-events: none;
        }

        .hero-kicker {
            font-size: 0.9rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            opacity: 0.78;
            margin-bottom: 10px;
        }

        .hero-title {
            font-size: 2.5rem;
            line-height: 1.08;
            font-weight: 800;
            margin: 0 0 12px 0;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: rgba(255,255,255,0.82);
            max-width: 900px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            border-radius: 999px;
            padding: 10px 16px;
            margin: 10px 0 18px 0;
            font-weight: 600;
            font-size: 0.96rem;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 12px 30px rgba(15,23,42,0.06);
        }

        .status-pill.success {
            background: var(--success-bg);
            color: #166534;
        }

        .status-pill.warn {
            background: var(--warn-bg);
            color: #92400e;
        }

        .section-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 24px 24px 10px 24px;
            box-shadow: 0 12px 30px rgba(15,23,42,0.06);
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }

        .section-card h2, .section-card h3 {
            margin-top: 0;
        }

        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 24px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 14px 28px rgba(15,23,42,0.06);
        }

        [data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: #111827;
            font-weight: 800;
        }

        .stPlotlyChart, .stDataFrame, [data-testid="stImage"] {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 24px;
            padding: 10px;
            box-shadow: 0 14px 28px rgba(15,23,42,0.05);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.65);
            border-radius: 14px 14px 0 0;
            padding-left: 16px;
            padding-right: 16px;
        }

        .stAlert {
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.16);
            box-shadow: 0 10px 24px rgba(15,23,42,0.05);
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(37,99,235,0.14);
            background: linear-gradient(135deg, rgba(37,99,235,0.96), rgba(124,58,237,0.92));
            color: white;
            font-weight: 700;
            box-shadow: 0 12px 22px rgba(37,99,235,0.22);
        }

        .stTextArea textarea, .stTextInput input, .stNumberInput input {
            border-radius: 14px;
        }

        hr {
            margin-top: 1.6rem !important;
            margin-bottom: 1.3rem !important;
            border-color: rgba(148,163,184,0.18);
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }

        .sidebar-chatbot-card {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 22px;
            padding: 18px 16px;
            background: linear-gradient(135deg, rgba(37,99,235,0.20), rgba(124,58,237,0.22));
            box-shadow: 0 14px 30px rgba(15,23,42,0.18);
            text-align: center;
            margin-bottom: 10px;
        }

        .sidebar-chatbot-emoji {
            font-size: 3rem;
            line-height: 1;
            margin-bottom: 10px;
        }

        .sidebar-chatbot-title {
            color: #ffffff;
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .sidebar-chatbot-desc {
            color: rgba(229,238,252,0.84);
            font-size: 0.90rem;
            line-height: 1.45;
        }

        .chatbot-view-chip {
            display: inline-block;
            margin-top: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: #dbeafe;
            font-size: 0.80rem;
            font-weight: 600;
        }

        .chatbot-dialog-note {
            background: rgba(37,99,235,0.08);
            border: 1px solid rgba(37,99,235,0.12);
            border-radius: 14px;
            padding: 10px 12px;
            color: #334155;
            margin-bottom: 12px;
        }

        .chatbot-drag-handle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
            padding: 10px 12px;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(37,99,235,0.92));
            color: #ffffff;
            font-weight: 700;
            cursor: move;
            user-select: none;
        }

        .chatbot-drag-handle small {
            color: rgba(255,255,255,0.78);
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">Retention Intelligence Suite</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(message: str, variant: str = "success"):
    st.markdown(
        f'<div class="status-pill {variant}">{message}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_app_data():
    return load_dashboard_bundle()


@st.cache_data(show_spinner=False)
def load_training_artifacts_api(rebuild: bool = False):
    return fetch_training_artifacts(rebuild=rebuild)


@st.cache_data(show_spinner=False)
def load_saved_results_artifacts_api(budget: int, rebuild: bool = False):
    return fetch_saved_results_artifacts(budget=budget, rebuild=rebuild)


def _artifact_frame(records) -> pd.DataFrame:
    return pd.DataFrame(records or [])


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


def get_chat_history_key(view_key: str) -> str:
    return f"llm_chat_history_{view_key}"


def get_chat_input_key(view_key: str) -> str:
    return f"llm_chat_input_{view_key}"


def resolve_chatbot_image() -> Optional[str]:
    candidates = [
        Path(__file__).resolve().parent / "assets" / "chatbot.png",
        Path(__file__).resolve().parent / "assets" / "chatbot.jpg",
        Path(__file__).resolve().parent / "data" / "chatbot.png",
        Path(__file__).resolve().parent / "data" / "chatbot.jpg",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def close_llm_chat_dialog():
    st.session_state["llm_chat_open"] = False
    st.session_state["llm_chat_view_key"] = None


def build_contextual_chat_question(
    view_title: str,
    history: list,
    latest_question: str,
    max_messages: int = 6,
) -> str:
    recent_history = history[-max_messages:] if history else []
    if not recent_history:
        return latest_question

    history_lines = []
    for item in recent_history:
        role = "사용자" if item.get("role") == "user" else "AI"
        content = str(item.get("content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")

    if not history_lines:
        return latest_question

    history_block = "\\n".join(history_lines)
    return (
        f"현재 대시보드 화면: {view_title}\\n"
        "아래는 직전 대화 맥락이다. 반드시 이 맥락을 참고해 이어서 답변하라.\\n\\n"
        f"{history_block}\\n\\n"
        f"현재 질문: {latest_question}"
    )


def render_llm_summary(
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
    st.caption("추가 질문은 사이드바의 AI 챗봇 버튼을 눌러 이어서 대화할 수 있습니다.")


def render_sidebar_chatbot_launcher(
    view_key: str,
    view_title: str,
    llm_enabled: bool,
    api_key: Optional[str],
):
    st.divider()
    st.subheader("AI 챗봇")

    chatbot_image_path = resolve_chatbot_image()
    if chatbot_image_path:
        st.image(chatbot_image_path, use_container_width=True)
        st.caption("현재 화면의 표·그래프를 바탕으로 대화를 이어갈 수 있습니다.")
    else:
        st.markdown(
            f"""
            <div class="sidebar-chatbot-card">
                <div class="sidebar-chatbot-emoji">🤖</div>
                <div class="sidebar-chatbot-title">AI 분석 챗봇</div>
                <div class="sidebar-chatbot-desc">
                    현재 보고 있는 표·그래프를 함께 보면서
                    질문을 이어갈 수 있습니다.
                </div>
                <div class="chatbot-view-chip">{view_title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ready, status_message = get_llm_status(api_key)

    if st.button(
        "🤖 AI 챗봇 열기",
        key=f"open_chatbot_{view_key}",
        use_container_width=True,
        disabled=(not llm_enabled) or (not ready),
    ):
        st.session_state["llm_chat_open"] = True
        st.session_state["llm_chat_view_key"] = view_key
        st.rerun()

    if not llm_enabled:
        st.caption("LLM 기능이 꺼져 있어 챗봇을 열 수 없습니다.")
    elif not ready:
        pass
    else:
        st.caption("현재 화면 문맥을 유지한 채 질문할 수 있습니다.")


@st.dialog("AI 분석 챗봇")
def open_chatbot_dialog(
    view_key: str,
    view_title: str,
    payload: Dict,
    api_key: Optional[str],
    model_name: str,
):
    ready, status_message = get_llm_status(api_key)
    payload_json = build_payload_json(payload)
    history_key = get_chat_history_key(view_key)
    input_key = get_chat_input_key(view_key)

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    st.markdown(
        """
        <div id="chatbot-drag-handle" class="chatbot-drag-handle">
            <span>🤖 AI 분석 챗봇</span>
            <small>드래그해서 이동</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="chatbot-dialog-note">
            <strong>현재 화면:</strong> {view_title}<br/>
            현재 화면의 지표·표·그래프 요약 컨텍스트를 바탕으로 답변합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_col1, top_col2 = st.columns([1, 1])
    if top_col1.button("대화 지우기", key=f"clear_chat_{view_key}", use_container_width=True):
        st.session_state[history_key] = []
        st.rerun()
    if top_col2.button("닫기", key=f"close_chat_{view_key}", use_container_width=True):
        close_llm_chat_dialog()
        st.rerun()

    if not ready:
        st.info(status_message)
        return

    history = st.session_state[history_key]

    if not history:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "안녕하세요. 현재 보고 있는 대시보드 화면을 기준으로 설명해드릴게요.\n\n"
                "- 왜 이 지표가 높거나 낮은지\n"
                "- 어떤 고객/세그먼트가 핵심인지\n"
                "- 지금 예산·threshold에서 무엇을 바꾸면 좋을지\n"
                "같은 질문을 이어서 해보세요."
            )

    for item in history:
        role = item.get("role", "assistant")
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(item.get("content", ""))

    prompt = st.chat_input(
        "현재 화면에 대해 질문하세요.",
        key=input_key,
    )

    if prompt:
        history.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        contextual_question = build_contextual_chat_question(
            view_title=view_title,
            history=history[:-1],
            latest_question=prompt,
        )

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI가 답변하는 중입니다..."):
                try:
                    answer = get_session_cached_answer(
                        view_title=view_title,
                        payload_json=payload_json,
                        question=contextual_question,
                        api_key=api_key or "",
                        model_name=model_name,
                    )
                except Exception as exc:
                    answer = f"AI 답변 생성 중 오류가 발생했습니다: {exc}"

            st.markdown(answer)

        history.append({"role": "assistant", "content": answer})
        st.session_state[history_key] = history



def inject_draggable_chat_dialog():
    components.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;

          function setupDraggableDialog() {
            const handle = doc.getElementById('chatbot-drag-handle');
            if (!handle) return;

            const dialog = handle.closest('[role="dialog"]');
            if (!dialog) return;
            if (dialog.dataset.dragBound === '1') return;

            dialog.dataset.dragBound = '1';
            dialog.style.position = 'fixed';
            dialog.style.margin = '0';
            dialog.style.transform = 'none';
            dialog.style.right = '24px';
            dialog.style.top = '92px';
            dialog.style.left = 'auto';
            dialog.style.width = 'min(460px, 92vw)';
            dialog.style.maxWidth = '92vw';
            dialog.style.maxHeight = '82vh';
            dialog.style.overflow = 'auto';
            dialog.style.zIndex = '999999';

            let dragging = false;
            let startX = 0;
            let startY = 0;
            let startLeft = 0;
            let startTop = 0;

            function clamp(value, minValue, maxValue) {
              return Math.min(Math.max(value, minValue), maxValue);
            }

            function onMouseMove(event) {
              if (!dragging) return;

              const nextLeft = startLeft + (event.clientX - startX);
              const nextTop = startTop + (event.clientY - startY);
              const maxLeft = Math.max(12, window.parent.innerWidth - dialog.offsetWidth - 12);
              const maxTop = Math.max(12, window.parent.innerHeight - dialog.offsetHeight - 12);

              dialog.style.left = clamp(nextLeft, 12, maxLeft) + 'px';
              dialog.style.top = clamp(nextTop, 12, maxTop) + 'px';
              dialog.style.right = 'auto';
            }

            function onMouseUp() {
              dragging = false;
              doc.removeEventListener('mousemove', onMouseMove);
              doc.removeEventListener('mouseup', onMouseUp);
            }

            handle.addEventListener('mousedown', function(event) {
              if (event.target.closest('button, input, textarea, a, label')) return;

              dragging = true;
              const rect = dialog.getBoundingClientRect();
              startLeft = rect.left;
              startTop = rect.top;
              startX = event.clientX;
              startY = event.clientY;

              dialog.style.left = rect.left + 'px';
              dialog.style.top = rect.top + 'px';
              dialog.style.right = 'auto';

              doc.addEventListener('mousemove', onMouseMove);
              doc.addEventListener('mouseup', onMouseUp);
              event.preventDefault();
            });
          }

          setupDraggableDialog();
          const observer = new MutationObserver(setupDraggableDialog);
          observer.observe(doc.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


inject_custom_css()

bundle = load_app_data()

customers = bundle.customer_summary
cohort_df = bundle.cohort_retention

render_hero(
    "AI 기반 고객 이탈 예측 및 리텐션 ROI 최적화 대시보드",
    "이탈 위험 고객 탐지부터 uplift·CLV·예산 최적화·개인화 추천까지 한 화면에서 이어서 분석하는 운영형 대시보드입니다.",
)

if bundle.used_mock:
    render_status_pill("실제 data/raw 산출물을 찾지 못해 mock data로 실행 중입니다.", "warn")
elif bundle.source_dir:
    render_status_pill(f"실제 시뮬레이터 산출물 사용 중: {bundle.source_dir}", "success")

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
            "7. 학습 결과 아티팩트",
            "8. 저장된 Uplift/최적화 결과",
            "9. 개인화 추천",
        ],
    )

    threshold = 0.50
    budget = 5_000_000
    top_n = 20
    target_cap = 1000
    recommendation_per_customer = 3

    if view in {"1. 이탈현황", "4. 예산 배분 결과", "5. 예상 최적화 ROI", "6. 리텐션 대상 고객 목록", "9. 개인화 추천"}:
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

    if view == "9. 개인화 추천":
        st.caption("최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다.")
        recommendation_per_customer = st.slider(
            "고객당 추천 개수",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )

    if view in {"4. 예산 배분 결과", "5. 예상 최적화 ROI", "9. 개인화 추천"}:
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

    if view == "9. 개인화 추천":
        preview_selected_customers, _, _ = get_budget_result(
            customers,
            budget=budget,
            threshold=threshold,
            max_customers=target_cap,
        )
        final_target_count = int(len(preview_selected_customers))
        top_n = int(st.number_input(
            "표시 고객 수",
            min_value=1,
            max_value=max(final_target_count, 1),
            value=min(20, max(final_target_count, 1)),
            step=1,
            help="최종 타겟 고객 수를 넘지 않는 범위에서 입력합니다.",
        ))
        st.caption(f"최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다. 현재 조건의 최종 타겟 고객 수: {final_target_count:,}명")

    st.divider()
    st.subheader("실행 / 새로고침")
    if st.button("데이터/결과 새로고침", use_container_width=True):
        load_app_data.clear()
        load_training_artifacts_api.clear()
        load_saved_results_artifacts_api.clear()
        st.rerun()

    st.caption(
        "기존 4·5번 화면은 저장 파일을 읽는 것이 아니라 현재 data/raw를 기준으로 다시 계산합니다."
    )

    st.divider()
    st.subheader("LLM 설정")
    st.caption(
        "권장: API 키는 코드에 쓰지 말고 환경변수 OPENAI_API_KEY 또는 Streamlit secrets로 관리하세요."
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

    render_sidebar_chatbot_launcher(
        view_key=view.split(".")[0],
        view_title=view,
        llm_enabled=llm_enabled,
        api_key=llm_api_key.strip() if llm_api_key else None,
    )


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

if view == "9. 개인화 추천":
    try:
        recommendation_summary, personalized_recommendations = fetch_personalized_recommendations(
            limit=top_n,
            per_customer=recommendation_per_customer,
            budget=budget,
            threshold=threshold,
            max_customers=target_cap,
            rebuild=True,
        )
    except Exception as exc:
        recommendation_summary, personalized_recommendations = {}, pd.DataFrame()
        recommendation_error = str(exc)
    else:
        recommendation_error = None
else:
    recommendation_summary, personalized_recommendations = {}, pd.DataFrame()
    recommendation_error = None

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
        hist_fig.update_traces(
            marker_line_color="rgba(255,255,255,0.95)",
            marker_line_width=1.2,
            opacity=0.9,
        )

        hist_fig.update_layout(
            bargap=0.02,
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
        "그래야 최근 코호트가 오른쪽 검열 때문에 과소평가되지 않습니다."
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
        "버블 크기는 expected_incremental_profit 대신 value_score(CLV × uplift_score)를 사용합니다."
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
    st.caption("이 화면은 저장된 optimize 결과 파일이 아니라 현재 입력값으로 다시 계산한 결과입니다.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 예산", money(optimize_summary["budget"]))
    m2.metric("집행 예산", money(optimize_summary["spent"]))
    m3.metric("잔여 예산", money(optimize_summary["remaining"]))
    m4.metric("타겟 고객 수", f"{optimize_summary['num_targeted']:,}")

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
        "selected_customer_numeric_summary": numeric_summary(
            selected_customers, ["coupon_cost", "expected_incremental_profit", "expected_roi"]
        ),
    }

elif view == "5. 예상 최적화 ROI":
    st.subheader("예상 최적화 ROI")
    st.caption("이 화면도 현재 입력값 기준의 실시간 재계산 결과입니다.")

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
        roi_fig.update_traces(
            marker_line_color="rgba(255,255,255,0.95)",
            marker_line_width=1.2,
            opacity=0.9,
        )

        roi_fig.update_layout(
            bargap=0.02,
        )

        st.plotly_chart(roi_fig, use_container_width=True)

        top_roi = selected_customers.sort_values("expected_roi", ascending=False).head(
            min(20, len(selected_customers))
        )
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
    }

elif view == "6. 리텐션 대상 고객 목록":
    st.subheader("리텐션 대상 고객 목록")

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
    }

elif view == "7. 학습 결과 아티팩트":
    st.subheader("학습 결과 아티팩트")
    st.caption("이 화면은 API 서버가 로컬 results/, models/, data/feature_store 를 읽고 필요하면 생성한 뒤 전달합니다.")

    try:
        training_payload = load_training_artifacts_api(rebuild=False)
    except Exception as exc:
        st.error(f"학습 결과 API 호출 실패: {exc}")
        training_payload = {}

    churn_metrics = training_payload.get("churn_metrics", {})
    threshold_analysis = training_payload.get("threshold_analysis", {})
    top_feature_importance_df = _artifact_frame(training_payload.get("top_feature_importance"))
    customer_features_df = _artifact_frame(training_payload.get("customer_features"))
    image_paths = training_payload.get("image_paths", {})
    model_paths = training_payload.get("model_paths", {})

    if not churn_metrics:
        st.warning("학습 결과를 아직 불러오지 못했습니다.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best model", str(churn_metrics.get("best_model_name", "-")))
        m2.metric("Test AUC", f"{float(churn_metrics.get('test_auc_roc', 0.0)):.4f}")
        m3.metric("Selected threshold", f"{float(churn_metrics.get('selected_threshold', 0.0)):.4f}")
        m4.metric("Positive rate", f"{float(churn_metrics.get('positive_rate', 0.0)):.2%}")

        st.markdown("### 학습 메타데이터")
        meta_df = pd.DataFrame(
            [
                {"key": "train_rows", "value": churn_metrics.get("train_rows")},
                {"key": "test_rows", "value": churn_metrics.get("test_rows")},
                {"key": "numeric_feature_count", "value": churn_metrics.get("numeric_feature_count")},
                {"key": "categorical_feature_count", "value": churn_metrics.get("categorical_feature_count")},
                {"key": "lightgbm_available", "value": churn_metrics.get("lightgbm_available")},
                {"key": "model_path", "value": model_paths.get("churn_model")},
            ]
        )
        st.dataframe(meta_df, use_container_width=True, hide_index=True)

    if not top_feature_importance_df.empty:
        st.markdown("### Top feature importance")
        st.dataframe(top_feature_importance_df, use_container_width=True, hide_index=True)

    if threshold_analysis and threshold_analysis.get("selected"):
        st.markdown("### 선택된 threshold 요약")
        selected_df = pd.DataFrame([threshold_analysis["selected"]])
        st.dataframe(selected_df, use_container_width=True, hide_index=True)

    st.markdown("### 학습 시각화")
    image_cols = st.columns(2)
    image_items = [
        ("ROC Curve", image_paths.get("churn_auc_roc")),
        ("Precision-Recall Tradeoff", image_paths.get("churn_precision_recall_tradeoff")),
        ("SHAP Summary", image_paths.get("churn_shap_summary")),
        ("SHAP Local", image_paths.get("churn_shap_local")),
    ]
    for idx, (title, img_path) in enumerate(image_items):
        with image_cols[idx % 2]:
            if img_path:
                st.image(img_path, caption=title, use_container_width=True)
            else:
                st.info(f"{title} 파일이 없습니다.")

    if not customer_features_df.empty:
        st.markdown("### Feature store 미리보기")
        st.dataframe(
            customer_features_df.head(20),
            use_container_width=True,
            hide_index=True,
        )

    llm_payload = {
        "churn_metrics": churn_metrics,
        "threshold_analysis_selected": threshold_analysis.get("selected", {}) if threshold_analysis else {},
        "top_feature_importance": top_feature_importance_df.to_dict(orient="records") if not top_feature_importance_df.empty else [],
        "feature_store_preview": dataframe_snapshot(
            customer_features_df,
            columns=list(customer_features_df.columns[:12]),
            max_rows=10,
        ) if not customer_features_df.empty else [],
    }

elif view == "8. 저장된 Uplift/최적화 결과":
    st.subheader("저장된 Uplift/최적화 결과")
    st.caption("이 화면은 API 서버가 로컬 results 산출물을 읽고, 없으면 현재 data/raw 기준으로 생성한 뒤 전달합니다.")

    try:
        saved_payload = load_saved_results_artifacts_api(int(budget), rebuild=False)
    except Exception as exc:
        st.error(f"저장 결과 API 호출 실패: {exc}")
        saved_payload = {}

    uplift_summary = saved_payload.get("uplift_summary", {})
    uplift_segmentation_df = _artifact_frame(saved_payload.get("uplift_segmentation"))
    optimization_summary = saved_payload.get("optimization_summary", {})
    optimization_segment_budget_df = _artifact_frame(saved_payload.get("optimization_segment_budget"))
    optimization_selected_customers_df = _artifact_frame(saved_payload.get("optimization_selected_customers"))

    uplift_tab, optimize_tab = st.tabs(["Uplift 결과", "Optimize 결과"])

    with uplift_tab:
        if not uplift_summary and uplift_segmentation_df.empty:
            st.warning("저장된 uplift 결과를 찾지 못했습니다.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Uplift rows", int(uplift_summary.get("rows", len(uplift_segmentation_df))))
            segment_counts = uplift_summary.get("segment_counts", {})
            m2.metric("세그먼트 종류 수", len(segment_counts))

            if segment_counts:
                seg_df = pd.DataFrame(
                    {
                        "uplift_segment": list(segment_counts.keys()),
                        "customer_count": list(segment_counts.values()),
                    }
                )
                fig = px.bar(seg_df, x="uplift_segment", y="customer_count", text="customer_count")
                st.plotly_chart(fig, use_container_width=True)

            if not uplift_segmentation_df.empty:
                st.dataframe(
                    uplift_segmentation_df.head(30),
                    use_container_width=True,
                    hide_index=True,
                )

    with optimize_tab:
        if not optimization_summary and optimization_segment_budget_df.empty:
            st.warning("저장된 optimize 결과를 찾지 못했습니다.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("저장된 예산", money(optimization_summary.get("budget", 0)))
            m2.metric("저장된 집행 예산", money(optimization_summary.get("spent", 0)))
            m3.metric("저장된 잔여 예산", money(optimization_summary.get("remaining", 0)))
            m4.metric("저장된 타겟 고객 수", f"{int(optimization_summary.get('num_targeted', 0)):,}")

            if not optimization_segment_budget_df.empty:
                display_df = optimization_segment_budget_df.copy()
                if "allocated_budget" in display_df.columns:
                    display_df["allocated_budget"] = display_df["allocated_budget"].map(money)
                if "expected_profit" in display_df.columns:
                    display_df["expected_profit"] = display_df["expected_profit"].map(money)
                st.markdown("### 세그먼트별 저장 결과")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            if not optimization_selected_customers_df.empty:
                st.markdown("### 저장된 선정 고객")
                st.dataframe(
                    optimization_selected_customers_df.head(30),
                    use_container_width=True,
                    hide_index=True,
                )

    llm_payload = {
        "uplift_summary": uplift_summary,
        "optimization_summary": optimization_summary,
        "optimization_segment_budget": optimization_segment_budget_df.to_dict(orient="records") if not optimization_segment_budget_df.empty else [],
        "optimization_selected_preview": dataframe_snapshot(
            optimization_selected_customers_df,
            columns=list(optimization_selected_customers_df.columns[:12]),
            max_rows=12,
        ) if not optimization_selected_customers_df.empty else [],
    }

elif view == "9. 개인화 추천":
    st.subheader("최종 타겟 고객 대상 개인화 추천")
    st.caption("예산/임계값으로 선별된 최종 리텐션 타겟 고객에게만 추천을 생성합니다. 추천 점수는 구매 이력 + 최근 관심 + 세그먼트 인기 + 전역 인기를 혼합해 계산합니다.")

    if recommendation_error:
        st.error(f"추천 API 호출 실패: {recommendation_error}")
    elif personalized_recommendations.empty:
        st.warning("표시할 추천 결과가 없습니다. 현재 예산/임계값 조건에서 최종 타겟 고객이 없을 수 있습니다.")
    else:
        budget_context = recommendation_summary.get('budget_context', {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("추천 행 수", f"{recommendation_summary.get('rows', len(personalized_recommendations)):,}")
        m2.metric("커버 고객 수", f"{recommendation_summary.get('customers_covered', personalized_recommendations['customer_id'].nunique()):,}")
        m3.metric("고객당 추천 수", str(recommendation_summary.get('per_customer', recommendation_per_customer)))
        m4.metric("최종 타겟 고객 수", f"{budget_context.get('num_targeted', recommendation_summary.get('customers_covered', 0)):,}")

        category_counts = (
            personalized_recommendations.groupby('recommended_category', as_index=False)
            .agg(recommend_count=('customer_id', 'count'))
            .sort_values('recommend_count', ascending=False)
        )
        fig = px.bar(
            category_counts,
            x='recommended_category',
            y='recommend_count',
            title='추천 카테고리 분포',
        )
        st.plotly_chart(fig, use_container_width=True)

        display_df = personalized_recommendations.copy()
        if 'churn_probability' in display_df.columns:
            display_df['churn_probability'] = display_df['churn_probability'].map(lambda x: f"{x:.3f}")
        if 'uplift_score' in display_df.columns:
            display_df['uplift_score'] = display_df['uplift_score'].map(lambda x: f"{x:.3f}")
        if 'clv' in display_df.columns:
            display_df['clv'] = display_df['clv'].map(money)
        if 'expected_incremental_profit' in display_df.columns:
            display_df['expected_incremental_profit'] = display_df['expected_incremental_profit'].map(money)
        if 'coupon_cost' in display_df.columns:
            display_df['coupon_cost'] = display_df['coupon_cost'].map(money)
        if 'expected_roi' in display_df.columns:
            display_df['expected_roi'] = display_df['expected_roi'].map(lambda x: f"{x:.3f}")
        if 'recommendation_priority' in display_df.columns:
            display_df['recommendation_priority'] = display_df['recommendation_priority'].map(lambda x: f"{x:.3f}")
        if 'target_priority_score' in display_df.columns:
            display_df['target_priority_score'] = display_df['target_priority_score'].map(lambda x: f"{x:.3f}")
        if 'recommendation_score' in display_df.columns:
            display_df['recommendation_score'] = display_df['recommendation_score'].map(lambda x: f"{x:.3f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    llm_payload = {
        'recommendation_summary': recommendation_summary,
        'category_distribution': (
            personalized_recommendations['recommended_category'].value_counts().to_dict()
            if not personalized_recommendations.empty else {}
        ),
        'recommendation_preview': dataframe_snapshot(
            personalized_recommendations,
            columns=[
                'customer_id',
                'persona',
                'recommended_category',
                'recommendation_rank',
                'recommendation_score',
                'reason_tags',
            ],
            max_rows=20,
        ) if not personalized_recommendations.empty else [],
    }


if llm_enabled:
    current_view_key = view.split(".")[0]
    current_model_name = llm_model.strip() or DEFAULT_MODEL_NAME

    render_llm_summary(
        view_key=current_view_key,
        view_title=llm_view_title,
        payload=llm_payload,
        api_key=llm_api_key_value,
        model_name=current_model_name,
    )

    if (
        st.session_state.get("llm_chat_open", False)
        and st.session_state.get("llm_chat_view_key") == current_view_key
    ):
        open_chatbot_dialog(
            view_key=current_view_key,
            view_title=llm_view_title,
            payload=llm_payload,
            api_key=llm_api_key_value,
            model_name=current_model_name,
        )
        inject_draggable_chat_dialog()
