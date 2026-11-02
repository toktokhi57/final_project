# -*- coding: utf-8 -*-
"""
STEP 23. 등급별 의사결정 규칙 (승인/거절/금리차등)

배경:
- 지금까지는 A~E "분류"만 했지, 그 등급을 가지고 실제로 "승인할지 거절할지, 금리를
  얼마나 다르게 매길지"는 정하지 않았다. 이 스크립트는 15_1번에서 이미 정립한
  LGD(부도시손실률)x순이익률 프레임워크를 그대로 재사용해서, 등급별 기대손익(Expected
  Value)을 계산하고, 이를 근거로 승인/거절/금리가산 기준을 데이터 기반으로 도출한다.
  (임의로 "D·E는 거절"이라고 정하는 게 아니라, 숫자로 손익분기점을 계산해서 정한다)

방법 (15_1번과 동일한 가정 재사용):
- LGD = 0.45 (부도가 나면 대출금액의 45%를 못 돌려받는다고 가정, Basel 관행치)
- 순이익률 = 0.08 (정상 상환시 대출금액의 8%가 은행 순이익이라고 가정)
- 대출 1건의 기대값(EV) = (1-p_부도) x 순이익률 x 대출금액 - p_부도 x LGD x 대출금액
  = 대출금액 x [(1-p_부도) x 0.08 - p_부도 x 0.45]
- EV=0이 되는 손익분기 부도율(breakeven PD)을 구해서, 등급별 실제부도율과 비교
- 위험조정 가산금리(risk premium) = LGD x p_부도 / (1-p_부도)
  (기대손실을 커버하기 위한 최소 가산금리 - 신용리스크 프라이싱의 표준 공식)

절차:
  1) 18번(등급변환_방법3_기저율중심.csv)의 등급별 실제부도율 로드
  2) 19번(등급별_프로파일링_수치형.csv)의 등급별 평균대출금액 로드
  3) 등급별 EV, 위험조정 가산금리 계산
  4) 손익분기 부도율(breakeven PD) 계산 및 각 등급과 비교
  5) EV 크기 기준으로 승인/거절/금리차등 규칙을 자동 생성 (임계값은 스크립트 상단
     THIN_MARGIN_RATIO로 조정 가능)

입력: 등급변환_방법3_기저율중심.csv(18번), 등급별_프로파일링_수치형.csv(19번)
출력: 등급별_의사결정규칙.csv, 등급별_기대손익_그래프.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
GRADES = ["A", "B", "C", "D", "E"]

LGD = 0.45          # 15_1번과 동일
NET_MARGIN = 0.08   # 15_1번과 동일

# 조건부승인(추가심사/한도축소) 판정 기준: A등급 EV 대비 이 비율 밑으로 얇아지면 "조건부"
THIN_MARGIN_RATIO = 0.30
STANDARD_RATIO = 0.70


def decide(ev, ev_a):
    if ev < 0:
        return "거절"
    elif ev < ev_a * THIN_MARGIN_RATIO:
        return "조건부승인(한도축소/가산금리 필수)"
    elif ev < ev_a * STANDARD_RATIO:
        return "표준승인"
    else:
        return "승인+우대금리"


if __name__ == "__main__":
    # =====================================================================
    # 1) 등급별 실제부도율 + 평균대출금액 로드
    # =====================================================================
    grade_rate = pd.read_csv(f"{OUT}/등급변환_방법3_기저율중심.csv", encoding=ENC, index_col=0)
    profile = pd.read_csv(f"{OUT}/등급별_프로파일링_수치형.csv", encoding=ENC, index_col=0)

    df = pd.DataFrame(index=GRADES)
    df["실제부도율"] = grade_rate.loc[GRADES, "실제부도율(%)"] / 100
    df["평균대출금액"] = profile.loc[GRADES, "대출금액_mean"]

    # =====================================================================
    # 2) EV(기대손익) 및 위험조정 가산금리 계산
    # =====================================================================
    p = df["실제부도율"]
    df["EV_비율"] = (1 - p) * NET_MARGIN - p * LGD
    df["EV_금액"] = df["EV_비율"] * df["평균대출금액"]
    df["위험조정가산금리(%p)"] = (LGD * p / (1 - p)) * 100

    breakeven_pd = NET_MARGIN / (NET_MARGIN + LGD)
    print(f"손익분기 부도율(breakeven PD) = {breakeven_pd:.4%}  "
          f"(LGD={LGD}, 순이익률={NET_MARGIN} 가정 기준)")
    print("-> 실제부도율이 이보다 높은 등급은 평균적으로 대출해줄수록 손해\n")

    # =====================================================================
    # 3) 의사결정 규칙 부여
    # =====================================================================
    ev_a = df.loc["A", "EV_금액"]
    df["권장조치"] = [decide(ev, ev_a) for ev in df["EV_금액"]]

    print("=== 등급별 의사결정 규칙 ===")
    print(df.round(4).to_string())

    df.to_csv(f"{OUT}/등급별_의사결정규칙.csv", encoding=ENC)

    # =====================================================================
    # 4) 시각화
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in df["EV_금액"]]
    ax.bar(GRADES, df["EV_금액"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("등급별 대출 1건당 기대손익(EV, 원)")
    ax.set_xlabel("등급")
    ax.set_ylabel("기대손익(원)")

    ax = axes[1]
    ax.bar(GRADES, df["위험조정가산금리(%p)"], color="#4a90d9")
    ax.set_title("등급별 위험조정 최소 가산금리(%p)")
    ax.set_xlabel("등급")
    ax.set_ylabel("가산금리(%p)")

    plt.tight_layout()
    plt.savefig(f"{OUT}/등급별_기대손익_그래프.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n저장 완료: 등급별_의사결정규칙.csv, 등급별_기대손익_그래프.png")
