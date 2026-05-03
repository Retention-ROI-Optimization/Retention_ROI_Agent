
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    ChatPromptTemplate = None
    StrOutputParser = None

DEFAULT_MODEL_NAME = "gpt-4.1-mini"


def _to_builtin(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable builtins."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return str(value)


def dataframe_snapshot(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    max_rows: int = 10,
    round_digits: int = 4,
) -> Dict[str, Any]:
    """Compact dataframe preview for LLM context."""
    target = df.copy()
    if columns is not None:
        existing = [col for col in columns if col in target.columns]
        target = target[existing]

    numeric_cols = target.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        target[numeric_cols] = target[numeric_cols].round(round_digits)

    sample = target.head(max_rows)
    return {
        "row_count": int(len(df)),
        "columns": list(target.columns),
        "sample_rows": [_to_builtin(row) for row in sample.to_dict(orient="records")],
    }


def series_distribution(df: pd.DataFrame, column: str, max_rows: int = 10) -> list[dict[str, Any]]:
    if column not in df.columns or df.empty:
        return []
    counts = (
        df[column]
        .value_counts(dropna=False)
        .head(max_rows)
        .rename_axis(column)
        .reset_index(name="count")
    )
    return [_to_builtin(row) for row in counts.to_dict(orient="records")]


def numeric_summary(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary[col] = {
            "min": round(float(series.min()), 4),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "max": round(float(series.max()), 4),
        }
    return summary


def build_payload_json(payload: Dict[str, Any]) -> str:
    safe = _to_builtin(payload)
    return json.dumps(safe, ensure_ascii=False, indent=2)


def get_api_key(user_api_key: Optional[str] = None) -> Optional[str]:
    if user_api_key and user_api_key.strip():
        return user_api_key.strip()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    return env_key or None


def get_llm_status(user_api_key: Optional[str] = None) -> tuple[bool, str]:
    if not LANGCHAIN_AVAILABLE:
        return (
            False,
            "LangChain 의존성이 아직 설치되지 않았습니다. `pip install -r requirements.txt`로 설치한 뒤 다시 실행하세요.",
        )
    if not get_api_key(user_api_key):
        return (
            False,
            "OpenAI API 키가 설정되지 않았습니다. 사이드바에 키를 입력하거나 `OPENAI_API_KEY` 환경변수를 설정하세요.",
        )
    return True, "ready"


def _build_llm(model_name: str, api_key: str):
    return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2, max_retries=2)


def generate_dashboard_summary(
    view_title: str,
    payload_json: str,
    user_api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    ready, message = get_llm_status(user_api_key)
    if not ready:
        raise RuntimeError(message)

    api_key = get_api_key(user_api_key)
    llm = _build_llm(model_name=model_name, api_key=api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
너는 리텐션 ROI 대시보드 분석 보조 AI다.
반드시 제공된 데이터만 근거로 답한다.
출력은 한국어로 작성한다.
추측하지 말고, 데이터에 없는 내용은 없다고 명시한다.
과장된 마케팅 문구 대신 분석가처럼 간결하게 쓴다.
다음 형식으로 답한다.

[핵심 결론]
- 2~4문장으로 가장 중요한 결론 요약

[주요 포인트]
- 3개 이하 bullet

[권장 액션]
- 1~2개 bullet
""".strip(),
            ),
            (
                "human",
                """
현재 화면: {view_title}

대시보드 데이터:
{payload_json}

위 데이터를 바탕으로 이 화면의 수치, 그래프, 표가 의미하는 바를 요약해줘.
비율/금액/증감 방향을 구체적으로 짚어줘.
""".strip(),
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"view_title": view_title, "payload_json": payload_json}).strip()


def answer_dashboard_question(
    view_title: str,
    payload_json: str,
    question: str,
    user_api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    ready, message = get_llm_status(user_api_key)
    if not ready:
        raise RuntimeError(message)

    api_key = get_api_key(user_api_key)
    llm = _build_llm(model_name=model_name, api_key=api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
너는 리텐션 ROI 대시보드용 질의응답 AI다.
반드시 제공된 대시보드 데이터만 사용해 답한다.
근거가 부족하면 "현재 화면 데이터만으로는 확정할 수 없습니다"라고 말한다.
답변은 한국어로, 직접적이고 실무적으로 쓴다.
가능하면 숫자 근거를 포함한다.
""".strip(),
            ),
            (
                "human",
                """
현재 화면: {view_title}
질문: {question}

대시보드 데이터:
{payload_json}

위 데이터만 바탕으로 질문에 답해줘.
""".strip(),
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {"view_title": view_title, "question": question, "payload_json": payload_json}
    ).strip()
