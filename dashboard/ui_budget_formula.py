"""Budget optimization formula explanations shared by the dashboard UI."""
from __future__ import annotations

from dashboard.ui_labels import translate_text


def budget_formula_html(lang: str = "ko") -> str:
    title = translate_text("예산·이익·ROI 산출식", lang)
    profit = translate_text("예상 추가 이익 = 고객 가치 × 개입 반응 가능성 × 이탈 위험도 - 개입 비용", lang)
    roi = translate_text("예상 ROI = 예상 추가 이익 ÷ 개입 비용", lang)
    note = translate_text("예산 최적화는 예상 추가 이익과 ROI가 높은 고객부터 예산 한도 안에서 선택합니다.", lang)
    return f"""
<div style="background:#F8FAFC;border:1px solid #CBD5E1;border-radius:14px;padding:14px 16px;margin:10px 0 18px 0;line-height:1.65;color:#0F172A;">
  <div style="font-weight:800;margin-bottom:6px;">📌 {title}</div>
  <div><b>1)</b> {profit}</div>
  <div><b>2)</b> {roi}</div>
  <div style="color:#64748B;font-size:13px;margin-top:6px;">{note}</div>
</div>
"""
