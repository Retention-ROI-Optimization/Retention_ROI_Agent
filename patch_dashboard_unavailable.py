"""
patch_dashboard_unavailable.py
==============================

Retention_ROI_Agent_customer-data 레포의 dashboard/app.py 를
자동으로 패치한다. 자사 데이터(user) 모드일 때
Treatment/Control 정보가 필요한 6개 화면을
"🔒 해당 데이터 없음" 박스로 깔끔하게 교체한다.

대상 화면:
  ③ Uplift + CLV 상위 고객
  ④ 예산 배분 결과
  ⑤ 예상 최적화 ROI
  ⑥ 리텐션 대상 고객 목록
  ⑧ Uplift/최적화 결과 (실시간)
  ⑭ 증분 성과 / A-B 실험

이 패치는:
  1) dashboard/app.py 상단에 헬퍼 함수 _user_mode_unavailable() 를 주입한다.
  2) 위 6개 화면의 elif 블록 시작 직후에 가드 한 줄을 추가한다.
     (자사 모드면 즉시 박스 표시 + return 처리)

사용 방법 (Windows CMD)
-----------------------
   py patch_dashboard_unavailable.py

복원 (만약 결과가 이상하다면):
   copy dashboard\\app.py.backup dashboard\\app.py

이 스크립트는 안전장치로 다음을 한다:
  - 패치 적용 전 dashboard/app.py.backup_before_patch 자동 백업
  - 이미 패치가 적용되어 있으면 건너뜀 (중복 방지)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


APP_PY = Path("dashboard/app.py")
BACKUP_PATH = Path("dashboard/app.py.backup_before_patch")

# 헬퍼 함수: dashboard/app.py 상단(import 끝나는 자리)에 삽입
HELPER_FUNCTION = '''
# ============================================================
# [PATCH] 자사 데이터(user) 모드에서 Treatment/Control 의존
# 화면을 "해당 데이터 없음" 으로 처리하는 헬퍼.
# 외부 데이터(UCI / Retailrocket 등)에는 처치/대조 정보가 없어
# Uplift, A/B 테스트, 예산 최적화 등을 산출할 수 없기 때문.
# ============================================================
def _user_mode_unavailable(feature_name: str, reason: str = "") -> bool:
    """현재 자사 데이터 모드면 '데이터 없음' 박스를 그리고 True 반환.
    시뮬레이터 모드면 False 반환 (원래 화면 그대로 진행)."""
    import streamlit as _st
    _mode = _st.session_state.get("data_mode", "simulator")
    if _mode != "user":
        return False
    _default_reason = (
        "외부 자사 데이터에는 Treatment/Control(처치·대조군) 배정 정보가 "
        "없어 이 항목은 산출할 수 없습니다."
    )
    _reason = reason or _default_reason
    _st.markdown(
        f"""
        <div style="
            background-color: #F3F4F6;
            border: 1px dashed #9CA3AF;
            border-radius: 12px;
            padding: 32px 24px;
            margin: 16px 0;
            text-align: center;
        ">
            <div style="font-size: 40px; opacity: 0.5;">🔒</div>
            <div style="font-size: 20px; font-weight: 700; color: #374151; margin-top: 8px;">
                해당 데이터 없음
            </div>
            <div style="font-size: 14px; color: #6B7280; margin-top: 8px;">
                {feature_name}
            </div>
            <div style="font-size: 13px; color: #9CA3AF; margin-top: 12px; line-height: 1.5;">
                {_reason}
            </div>
            <div style="font-size: 12px; color: #9CA3AF; margin-top: 12px; font-style: italic;">
                💡 사이드바에서 'Simulator 데모' 모드로 전환하면 확인할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return True
# ============================================================
# [/PATCH]
# ============================================================

'''

# 가드를 삽입할 view 분기들
# 각 항목: (elif 줄 정확한 매치, 화면 이름, 가드 사유)
VIEW_GUARDS = [
    (
        'elif view == "3. Uplift + CLV 상위 고객":',
        "Uplift Score + CLV 상위 고객 분석",
        "외부 자사 데이터에는 Treatment/Control 배정 정보가 없어 Uplift Score 계산이 불가합니다.",
    ),
    (
        'elif view == "4. 예산 배분 결과":',
        "예산 배분 결과 (세그먼트별 배분)",
        "예산 배분은 Uplift 세그먼트(Persuadables 등)를 기반으로 하며, 외부 데이터에는 Treatment 정보가 없어 산출 불가합니다.",
    ),
    (
        'elif view == "5. 예상 최적화 ROI":',
        "예상 최적화 ROI",
        "ROI 계산은 Uplift 기반 증분 이익 추정이 필요한데, 외부 데이터에는 Treatment 정보가 없어 산출 불가합니다.",
    ),
    (
        'elif view == "6. 리텐션 대상 고객 목록":',
        "리텐션 대상 고객 목록",
        "최종 리텐션 타겟 선정은 Uplift Score + 예산 최적화에 의존하며, 외부 데이터로는 산출 불가합니다.",
    ),
    (
        'elif view == "8. Uplift/최적화 결과 (실시간)":',
        "Uplift / 최적화 실시간 결과",
        "외부 자사 데이터에는 Treatment/Control 배정 정보가 없어 Uplift 학습과 최적화가 불가합니다.",
    ),
    (
        'elif view == "14. 증분 성과 / A-B 실험":',
        "증분 성과 / A-B 실험 분석",
        "A/B 테스트 분석은 Treatment/Control 그룹 분리 데이터가 필수이며, 외부 데이터에는 해당 정보가 없습니다.",
    ),
]

# 가드 코드 템플릿 (각 elif 직후에 삽입)
GUARD_TEMPLATE = '    if _user_mode_unavailable("{feature}", "{reason}"):\n        st.stop()\n'

# 패치 마커 (이미 패치됐는지 확인용)
PATCH_MARKER = "def _user_mode_unavailable("


def main() -> int:
    if not APP_PY.exists():
        print(f"[error] {APP_PY} 가 없습니다. 레포 루트에서 실행하세요.")
        print(f"        현재 위치: {Path.cwd()}")
        return 1

    content = APP_PY.read_text(encoding="utf-8")

    # 이미 패치된 경우
    if PATCH_MARKER in content:
        print(f"[skip] 이미 패치가 적용되어 있습니다.")
        print(f"       다시 적용하려면 백업에서 복원 후 실행하세요:")
        print(f"       copy dashboard\\app.py.backup_before_patch dashboard\\app.py")
        return 0

    # 백업
    BACKUP_PATH.write_text(content, encoding="utf-8")
    print(f"[backup] {BACKUP_PATH} 생성 완료")

    # ----------------------------------------------------------
    # 1단계: 헬퍼 함수 주입
    # streamlit 첫 import 직후 또는 첫 def/class 직전에 삽입
    # ----------------------------------------------------------
    # 가장 안전한 자리: 첫 번째 함수 정의(`def `) 직전
    match = re.search(r"^def ", content, flags=re.MULTILINE)
    if not match:
        print(f"[error] dashboard/app.py 에서 첫 함수 정의를 찾지 못했습니다.")
        return 1
    insert_pos = match.start()
    content = content[:insert_pos] + HELPER_FUNCTION + content[insert_pos:]
    print(f"[ok] 헬퍼 함수 _user_mode_unavailable 주입 완료 (위치: 첫 def 직전)")

    # ----------------------------------------------------------
    # 2단계: 6개 view 분기에 가드 삽입
    # ----------------------------------------------------------
    inserted_count = 0
    for elif_line, feature, reason in VIEW_GUARDS:
        if elif_line not in content:
            print(f"[warn] 분기를 찾지 못함: {elif_line}")
            continue

        # 안전한 reason 처리 (큰따옴표 escape)
        safe_reason = reason.replace('"', "'")
        guard_code = GUARD_TEMPLATE.format(feature=feature, reason=safe_reason)

        # 분기 줄 직후에 가드 코드 삽입
        # elif 줄 + 그 줄의 개행을 찾아서 그 다음에 가드 삽입
        pattern = re.escape(elif_line) + r"\n"
        replacement = elif_line + "\n" + guard_code

        new_content, n_subs = re.subn(pattern, replacement, content, count=1)
        if n_subs > 0:
            content = new_content
            inserted_count += 1
            print(f"[ok] 가드 삽입: {feature}")
        else:
            print(f"[warn] 가드 삽입 실패: {feature}")

    # ----------------------------------------------------------
    # 3단계: 저장
    # ----------------------------------------------------------
    APP_PY.write_text(content, encoding="utf-8")
    print(f"\n[done] 패치 완료. {inserted_count}/6 화면에 가드 적용됨.")
    print(f"\n다음 단계:")
    print(f"  1. 기존 streamlit 프로세스가 떠있으면 Ctrl+C 로 종료")
    print(f"  2. py -m streamlit run dashboard\\app.py")
    print(f"  3. 자사 데이터 모드에서 ③④⑤⑥⑧⑭ 화면 클릭해서 확인")
    print(f"\n복원이 필요하면:")
    print(f"  copy dashboard\\app.py.backup_before_patch dashboard\\app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
