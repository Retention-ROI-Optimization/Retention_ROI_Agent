"""Budget and ROI formula block used in the budget optimization view."""

from __future__ import annotations


def budget_formula_html(language_code: str = "ko") -> str:
    if language_code == "en":
        title = "Budget / Profit / ROI calculation"
        lines = [
            ("Expected incremental profit", "Customer value × response likelihood × churn risk − intervention cost"),
            ("Expected ROI", "Expected incremental profit ÷ intervention cost"),
            ("Recommended investment", "The per-customer intervention cost actually allocated to the action queue."),
        ]
        note = "Use this block to explain why a customer was selected and how much budget is needed."
    elif language_code == "ja":
        title = "予算・利益・ROIの算出式"
        lines = [
            ("予想追加利益", "顧客価値 × 介入反応見込み × 離脱リスク − 介入費用"),
            ("予想ROI", "予想追加利益 ÷ 介入費用"),
            ("推奨投資額", "アクションキューに実際に割り当てる顧客別介入費用です。"),
        ]
        note = "この式により、なぜ対象になったのか、いくら投資すべきかを説明できます。"
    else:
        title = "예산·이익·ROI 산출식"
        lines = [
            ("예상 추가 이익", "고객 가치 × 개입 반응 가능성 × 이탈 위험도 − 개입 비용"),
            ("예상 ROI", "예상 추가 이익 ÷ 개입 비용"),
            ("추천 투자액", "액션 큐에 실제로 배정되는 고객별 개입 비용입니다."),
        ]
        note = "이 산식을 통해 고객 선정 이유와 고객별 투자 필요 금액을 함께 설명할 수 있습니다."

    body = "".join(
        f"<li><strong>{label}</strong>: <code>{formula}</code></li>"
        for label, formula in lines
    )
    return f"""
<div style="border:1px solid #dbeafe;background:#eff6ff;border-radius:14px;padding:14px 16px;margin:8px 0 18px 0;">
  <div style="font-weight:800;color:#1e3a8a;margin-bottom:6px;">{title}</div>
  <ul style="margin:0 0 8px 18px;padding:0;color:#172554;">{body}</ul>
  <div style="font-size:0.9rem;color:#475569;">{note}</div>
</div>
"""
