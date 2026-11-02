# -*- coding: utf-8 -*-
"""
STEP 19. 등급별(A~E) 고객 특성 프로파일링

배경:
- 18번에서 확정한 방법3(기저율중심) 등급을 채택했다. 이 등급이 단순히 "위험도 순서"만
  잘 나눈 게 아니라, 실제로 어떤 종류의 고객들이 각 등급에 몰려있는지(소득, 대출금액,
  나이, 근속연수, 외부신용점수 등)를 보여줘야 발표/보고서에서 등급의 의미를 설명할 수 있다.

방법:
- 18번이 저장한 등급(방법3=기저율중심)을 valid셋 원본 피처(tree_encoded_valid.pkl)에 다시
  붙여서, 등급별로 주요 변수들의 평균/중앙값을 비교한다.
- 근속후경과일/출생후경과일은 원본이 "오늘 기준 며칠 전"을 음수로 저장한 값이라 그대로는
  읽기 어려우므로, 나이(세)/근속연수(년)로 변환해서 같이 보여준다.
- 스크립트에 적어둔 변수 후보 목록 중 실제로 존재하는 컬럼만 자동으로 골라서 사용하므로,
  컬럼명이 조금 다르더라도 에러 없이 있는 것만 프로파일링한다.

입력: tree_encoded_valid.pkl (08번 출력), 등급변환_valid셋_전체.csv (18번 출력)
출력: 등급별_프로파일링_수치형.csv, 등급별_프로파일링_박스플롯.png
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
GRADES = ["A", "B", "C", "D", "E"]
GRADE_COL = "등급_기저율중심"  # 최종 채택한 방법3

# 프로파일링 후보 변수 (실제 존재하는 것만 자동 선택)
CANDIDATE_COLS = [
    "대출금액", "총소득", "연간상환액", "구매물품가격",
    "외부신용점수_평균", "외부신용점수_최솟값", "외부신용점수1",
    "신용부담비율_상환대출", "계약금비율", "가족구성원수",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


if __name__ == "__main__":
    t0 = time.time()

    # =====================================================================
    # 1) 데이터 로드 및 등급 결합
    # =====================================================================
    log("=" * 70)
    log("valid셋 원본 피처 + 18번 등급 결과 로드")
    val = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl").reset_index(drop=True)
    grade_df = pd.read_csv(f"{OUT}/등급변환_valid셋_전체.csv", encoding=ENC).reset_index(drop=True)

    if len(val) != len(grade_df):
        raise ValueError(
            f"행 개수가 안 맞습니다 (val={len(val)}, grade_df={len(grade_df)}). "
            "17/18번을 다시 실행해서 tree_encoded_valid.pkl과 순서가 맞는 최신 등급변환_valid셋_전체.csv를 "
            "생성한 뒤 다시 시도하세요."
        )

    df = val.copy()
    df[GRADE_COL] = pd.Categorical(grade_df[GRADE_COL], categories=GRADES, ordered=True)
    df["실제_목표변수"] = grade_df["실제_목표변수"].values
    df["보정확률_Platt"] = grade_df["보정확률_Platt"].values

    # =====================================================================
    # 2) 나이(세) / 근속연수(년) 파생 (원본이 음수 일수로 저장돼 있어 변환)
    # =====================================================================
    derived_added = []
    if "출생후경과일" in df.columns:
        df["나이(세)"] = -df["출생후경과일"] / 365.25
        derived_added.append("나이(세)")
    if "근속후경과일" in df.columns:
        # Home Credit 원본 데이터의 무직자 sentinel 값(365243) 방어
        emp = df["근속후경과일"].where(df["근속후경과일"].abs() < 365243)
        df["근속연수(년)"] = -emp / 365.25
        derived_added.append("근속연수(년)")

    profile_cols = [c for c in CANDIDATE_COLS if c in df.columns] + derived_added
    if not profile_cols:
        raise ValueError("프로파일링할 수 있는 후보 컬럼이 하나도 없습니다. CANDIDATE_COLS를 "
                          "tree_encoded_valid.pkl의 실제 컬럼명에 맞게 수정하세요.")
    log(f"프로파일링 대상 변수({len(profile_cols)}개): {profile_cols}")

    # =====================================================================
    # 3) 등급별 평균/중앙값 요약
    # =====================================================================
    log("=" * 70)
    log("등급별 요약 통계 계산 중...")
    agg_funcs = ["mean", "median"]
    summary = df.groupby(GRADE_COL, observed=False)[profile_cols].agg(agg_funcs)
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary.insert(0, "인원수", df.groupby(GRADE_COL, observed=False).size())
    summary = summary.reindex(GRADES)
    print(summary.round(2).to_string())
    summary.to_csv(f"{OUT}/등급별_프로파일링_수치형.csv", encoding=ENC)

    # =====================================================================
    # 4) 박스플롯 (등급별 분포 시각화) - 대표 변수 최대 6개
    # =====================================================================
    log("=" * 70)
    log("박스플롯 생성 중...")
    plot_priority = ["외부신용점수_평균", "나이(세)", "근속연수(년)",
                      "대출금액", "총소득", "신용부담비율_상환대출"]
    plot_cols = [c for c in plot_priority if c in profile_cols]
    # 우선순위 목록에 없는데 존재하는 변수가 있으면 뒤에 채워서 최대 6개까지
    for c in profile_cols:
        if len(plot_cols) >= 6:
            break
        if c not in plot_cols:
            plot_cols.append(c)
    plot_cols = plot_cols[:6]

    n = len(plot_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        data_by_grade = [df.loc[df[GRADE_COL] == g, col].dropna() for g in GRADES]
        ax.boxplot(data_by_grade, tick_labels=GRADES, showfliers=False)
        ax.set_title(col)
        ax.set_xlabel("등급")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUT}/등급별_프로파일링_박스플롯.png", dpi=150, bbox_inches="tight")
    plt.close()

    log("=" * 70)
    log("저장 완료: 등급별_프로파일링_수치형.csv, 등급별_프로파일링_박스플롯.png")
    log(f"전체 소요시간: {time.time() - t0:.1f}초")
