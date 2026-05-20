"""LLM output-language policy for dashboard summaries and chatbot answers."""
from __future__ import annotations


def llm_language_name(lang: str = "ko") -> str:
    return {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(lang, "Korean")


def llm_language_instruction(lang: str = "ko") -> str:
    if lang == "en":
        return (
            "You must write the entire response in English. Do not write Korean or Japanese. "
            "Translate dashboard labels, table-cell explanations, and metric names into clear business English. "
            "Do not copy Korean or Japanese table values verbatim unless they are proper nouns."
        )
    if lang == "ja":
        return (
            "必ず回答全体を日本語で書いてください。韓国語や英語の文章を混ぜないでください。"
            "ダッシュボードのラベル、表のセル内説明、指標名も分かりやすい日本語に言い換えてください。"
            "固有名詞以外の韓国語・英語の値をそのまま引用しないでください。"
        )
    return (
        "반드시 전체 답변을 한국어로 작성하세요. 영어/일본어 문장을 섞지 말고, "
        "표·지표·추천 이유도 비즈니스 담당자가 이해하기 쉬운 한국어로 풀어 쓰세요."
    )
