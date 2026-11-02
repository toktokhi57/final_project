# -*- coding: utf-8 -*-
"""
STEP 18. A~E 신용등급 변환

배경:
- 17번에서 XGBoost 원본 확률(평균 36.3%, 실제 부도율 8.07%와 28.2%p 괴리)을
  Platt/Isotonic 두 방식으로 보정했고, 지표(AUC/Brier/평균확률-실제부도율 괴리) 전부에서
  Platt이 근소 우위 + 더 단순(과적합 위험 낮음)이라 최종 채택했다.
  이제 이 Platt 보정확률을 실제 신용등급(A=가장 안전 ~ E=가장 위험)으로 변환한다.

3가지 구간 산정 방식을 모두 계산해서 비교한다(사용자 선택: "세 개 다"):
  [방법1] 인원 등분할(quantile) - 보정확률을 크기순 정렬 후 정확히 20%씩 5등분.
          등급별 표본 수가 균등해 발표·통계 비교에 유리하지만, 등급 경계의 실제 확률값
          자체는 별 의미가 없을 수 있음(순전히 상대적 순위 기준).
  [방법2] 고정 확률구간(업계 관행형) - A<2%, B 2~5%, C 5~10%, D 10~20%, E>20%.
          등급의 확률적 의미가 고정되어 해석은 쉽지만, 우리 데이터 기저율(8.07%)과
          안 맞으면 등급별 인원이 극단적으로 쏠릴 수 있음.
  [방법3] 실제 기저율(8.07%) 중심 커스텀 - 우리 데이터의 실제 부도율을 C등급 중심에 두고,
          그 절반/두배 지점을 경계로 삼음: A<기저율/4, B<기저율/2, C<기저율, D<기저율*2, E>=기저율*2.
          업계 고정구간과 데이터 기저율 사이 절충안.

각 방법에 대해 등급별로 인원수·비율·평균보정확률·실제부도율(검증셋 실측)을 계산하고,
등급이 A->E로 갈수록 실제부도율이 단조증가하는지(=등급이 실제로 위험을 잘 구분하는지)도 확인한다.

입력: 17번이 저장한 calibrated_valid_proba_platt.npy, valid_y.npy (재학습 없이 재사용)
출력: 등급변환_방법1_인원균등분할.csv, 등급변환_방법2_고정확률구간.csv,
      등급변환_방법3_기저율중심.csv, 등급변환_3방법_비교.png,
      등급변환_valid셋_전체.csv (고객별 3가지 방법 등급 전부 포함)
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
GRADES = ["A", "B", "C", "D", "E"]  # A=가장 안전(저위험) ~ E=가장 위험(고위험)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def summarize(p, y, grade, method_name):
    """등급별 인원수/비율/평균보정확률/실제부도율 요약 + 단조성 체크"""
    df = pd.DataFrame({"보정확률_Platt": p, "실제_목표변수": y, "등급": grade})
    g = (
        df.groupby("등급", observed=True)
        .agg(
            인원수=("등급", "count"),
            확률_최소=("보정확률_Platt", "min"),
            확률_최대=("보정확률_Platt", "max"),
            평균보정확률=("보정확률_Platt", "mean"),
            부도자수=("실제_목표변수", "sum"),
        )
        .reindex(GRADES)
    )
    g["비율(%)"] = g["인원수"] / g["인원수"].sum() * 100
    g["실제부도율(%)"] = g["부도자수"] / g["인원수"] * 100
    g = g[["인원수", "비율(%)", "확률_최소", "확률_최대", "평균보정확률", "부도자수", "실제부도율(%)"]]

    rates = g["실제부도율(%)"].values
    is_monotonic = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    log(f"[{method_name}] 등급별 실제부도율 단조증가(A->E) 여부: "
        f"{'예 (등급이 위험을 잘 구분함)' if is_monotonic else '아니오 (일부 역전 있음 - 표본 부족 등 확인 필요)'}")
    print(g.round(4).to_string())
    return g


if __name__ == "__main__":
    t0 = time.time()

    # =====================================================================
    # 1) 17번 결과 로드 (Platt 보정확률 채택)
    # =====================================================================
    log("=" * 70)
    log("17번 결과 로드 (Platt 보정확률 + valid 실제값)")
    p = np.load(f"{OUT}/calibrated_valid_proba_platt.npy")
    y = np.load(f"{OUT}/valid_y.npy")
    base_rate = y.mean()
    log(f"valid셋 n={len(y)}, 실제 부도율(기저율)={base_rate:.4%}")

    result_df = pd.DataFrame({"보정확률_Platt": p, "실제_목표변수": y})

    # =====================================================================
    # [방법1] 인원 등분할 (quantile 5등분)
    # =====================================================================
    log("=" * 70)
    log("[방법1] 인원 등분할(quantile 5등분) 계산 중...")
    try:
        grade1 = pd.qcut(p, 5, labels=GRADES, duplicates="drop")
        if len(grade1.categories) < 5:
            raise ValueError("중복값으로 5개 구간이 안 나옴 -> 순위 기반으로 재시도")
    except Exception as e:
        log(f"  일반 qcut 실패({e}), 순위(rank) 기반으로 재계산")
        grade1 = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=GRADES)
    summary1 = summarize(p, y, grade1, "방법1-인원균등분할")
    summary1.to_csv(f"{OUT}/등급변환_방법1_인원균등분할.csv", encoding=ENC)
    result_df["등급_인원균등분할"] = np.asarray(grade1)

    # =====================================================================
    # [방법2] 고정 확률구간 (업계 관행형: A<2%, B 2~5%, C 5~10%, D 10~20%, E>=20%)
    # =====================================================================
    log("=" * 70)
    log("[방법2] 고정 확률구간(업계 관행형) 계산 중...")
    fixed_bins = [-0.001, 0.02, 0.05, 0.10, 0.20, 1.001]
    grade2 = pd.cut(p, bins=fixed_bins, labels=GRADES)
    log(f"  구간 경계: A<2%, B[2%,5%), C[5%,10%), D[10%,20%), E>=20%")
    summary2 = summarize(p, y, grade2, "방법2-고정확률구간")
    summary2.to_csv(f"{OUT}/등급변환_방법2_고정확률구간.csv", encoding=ENC)
    result_df["등급_고정확률구간"] = np.asarray(grade2)

    # =====================================================================
    # [방법3] 실제 기저율(8.07%) 중심 커스텀
    #   A < 기저율/4, B < 기저율/2, C < 기저율, D < 기저율*2, E >= 기저율*2
    # =====================================================================
    log("=" * 70)
    log("[방법3] 실제 기저율 중심 커스텀 구간 계산 중...")
    custom_bins = [-0.001, base_rate / 4, base_rate / 2, base_rate, base_rate * 2, 1.001]
    log(f"  구간 경계: A<{base_rate/4:.2%}, B<{base_rate/2:.2%}, C<{base_rate:.2%}, "
        f"D<{base_rate*2:.2%}, E>={base_rate*2:.2%}  (기저율={base_rate:.2%} 기준)")
    grade3 = pd.cut(p, bins=custom_bins, labels=GRADES)
    summary3 = summarize(p, y, grade3, "방법3-기저율중심")
    summary3.to_csv(f"{OUT}/등급변환_방법3_기저율중심.csv", encoding=ENC)
    result_df["등급_기저율중심"] = np.asarray(grade3)

    # =====================================================================
    # 전체 valid셋 3방법 등급 저장
    # =====================================================================
    result_df.to_csv(f"{OUT}/등급변환_valid셋_전체.csv", index=False, encoding=ENC)

    # =====================================================================
    # 시각화: 2행(인원분포 / 실제부도율) x 3열(방법1/2/3)
    # =====================================================================
    log("=" * 70)
    log("비교 시각화 생성 중...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    summaries = [("방법1: 인원균등분할", summary1), ("방법2: 고정확률구간", summary2),
                 ("방법3: 기저율중심", summary3)]

    for col, (title, s) in enumerate(summaries):
        ax_top = axes[0, col]
        ax_top.bar(GRADES, s["비율(%)"], color="#4a90d9")
        ax_top.set_title(title)
        ax_top.set_ylabel("인원 비율(%)" if col == 0 else "")
        ax_top.set_ylim(0, max(s["비율(%)"].max() * 1.2, 5))

        ax_bot = axes[1, col]
        ax_bot.bar(GRADES, s["실제부도율(%)"], color="#e94560")
        ax_bot.axhline(base_rate * 100, color="gray", linestyle="--", linewidth=1,
                        label=f"전체 기저율={base_rate:.2%}")
        ax_bot.set_ylabel("실제 부도율(%)" if col == 0 else "")
        ax_bot.set_xlabel("등급")
        ax_bot.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}/등급변환_3방법_비교.png", dpi=150, bbox_inches="tight")
    plt.close()

    log("=" * 70)
    log("저장 완료: 등급변환_방법1/2/3_*.csv, 등급변환_valid셋_전체.csv, 등급변환_3방법_비교.png")
    log(f"전체 소요시간: {time.time() - t0:.1f}초")
