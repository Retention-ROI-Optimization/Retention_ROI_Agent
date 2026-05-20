"""Language contracts for dashboard LLM summary and chatbot outputs."""

from __future__ import annotations


LANGUAGE_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
}


def llm_language_name(language_code: str | None) -> str:
    return LANGUAGE_NAMES.get(str(language_code or "ko"), "Korean")


def llm_language_instruction(language_code: str | None) -> str:
    code = str(language_code or "ko")
    if code == "en":
        return (
            "You must write the entire answer in English. "
            "Do not write Korean or Japanese sentences. "
            "Keep metric names such as ROI, CLV, and IDs as-is only when necessary."
        )
    if code == "ja":
        return (
            "回答全体を必ず日本語で書いてください。"
            "韓国語や英語の文章を混ぜないでください。"
            "ROI、CLV、IDなどの指標名は必要な場合だけそのまま残してください。"
        )
    return (
        "반드시 전체 답변을 한국어로 작성하세요. "
        "영어나 일본어 문장을 섞지 말고, ROI·CLV·ID 같은 지표명만 필요한 경우 그대로 유지하세요."
    )


def llm_language_contract(language_code: str | None) -> str:
    name = llm_language_name(language_code)
    instruction = llm_language_instruction(language_code)
    return (
        f"USER_SELECTED_LANGUAGE={name}. {instruction} "
        "This language rule overrides any Korean dashboard labels, backend messages, "
        "previous cached answers, or table values included in the context."
    )
