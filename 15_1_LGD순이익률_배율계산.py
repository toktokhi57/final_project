# -*- coding: utf-8 -*-
"""
STEP 14-1. LGD(부도시손실률) x 순이익률 기반 FN:FP 배율 계산
- 목적: 15번(이자계산/단순근사 방식)보다 금융권에서 더 표준적으로 쓰는 두 지표로
  FN:FP 배율을 다시 계산한다. 15번의 "대출 원금 전액 손실" 가정은 사실 다소 보수적이지
  않은(과장된) 가정이었다 - 실제로는 부도가 나도 담보처분·추심 등으로 원금의 일부는
  회수되기 때문이다.

  - LGD(Loss Given Default, 부도시손실률): 부도가 나도 원금 전체가 아니라 원금 중
    일부(LGD 비율)만큼만 손실로 잡는다. 무담보 개인신용대출은 Basel 기준 통상 45%를
    표준값으로 쓴다(비교적 안정적이고 근거 있는 가정).
      -> FN 손실 = LGD × 부도자(TARGET=1) 평균 대출금액

  - 순이익률: 정상 상환된 대출 1건에서 은행이 실제로 남기는 순이익 비율(이자수익에서
    대손비용·운영비·조달비용을 다 제외한 값). 소비자금융업계 평균은 통상 5~10%로 알려져
    있고 15%는 다소 높은 편이라, 8%를 보수적이면서도 비현실적이지 않은 기본값으로 쓴다.
      -> FP 손실 = 순이익률 × 정상고객(TARGET=0) 평균 대출금액

  - 배율 = FN 손실 / FP 손실

  LGD=0.45(업계 표준값), 순이익률=0.08(보수적 가정)이 가장 방어하기 쉬운 조합이라
  기본값으로 설정했다. 아래 LGD/NET_MARGIN 값을 바꾸면 다른 가정으로 다시 계산할 수 있다.

입력: full_feature_table_derived.pkl (07번 출력),
      xgb_valid_proba.npy / xgb_valid_y.npy (14번 출력 - 없으면 14번을 먼저 실행)
출력: LGD순이익률_배율_계산결과.csv, LGD순이익률_기반_임계값_결과.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"

LGD = 0.45         # 부도시손실률 (Basel 무담보 개인신용대출 표준값)
NET_MARGIN = 0.08  # 순이익률 (소비자금융업계 통상 5~10%, 보수적 기본값)


def log(msg):
    print(msg)


if __name__ == "__main__":
    df = pd.read_pickle(f"{OUT}/full_feature_table_derived.pkl")
    train = df[df["IS_TRAIN"] == 1].copy()
    del df

    credit = train["대출금액"]
    target = train[TARGET].astype(int)

    avg_credit_default = credit[target == 1].mean()
    avg_credit_normal = credit[target == 0].mean()
    log(f"부도자 평균 대출금액 = {avg_credit_default:,.0f}")
    log(f"정상고객 평균 대출금액 = {avg_credit_normal:,.0f}")

    fn_loss = LGD * avg_credit_default
    fp_loss = NET_MARGIN * avg_credit_normal
    ratio = fn_loss / fp_loss

    log(f"FN 손실 = LGD({LGD}) x 부도자평균대출금액 = {fn_loss:,.0f}")
    log(f"FP 손실 = 순이익률({NET_MARGIN}) x 정상고객평균대출금액 = {fp_loss:,.0f}")
    log("=" * 60)
    log(f"배율 = {ratio:.2f} : 1")

    pd.DataFrame([{
        "LGD": LGD, "순이익률": NET_MARGIN,
        "부도자_평균대출금액": round(avg_credit_default, 0),
        "정상고객_평균대출금액": round(avg_credit_normal, 0),
        "FN_손실": round(fn_loss, 0), "FP_손실": round(fp_loss, 0),
        "배율": round(ratio, 2),
    }]).to_csv(f"{OUT}/LGD순이익률_배율_계산결과.csv", index=False, encoding=ENC)

    # =====================================================================
    # 계산된 배율로 최적 임계값 탐색 (14/15번과 동일 로직)
    # =====================================================================
    proba_path = f"{OUT}/xgb_valid_proba.npy"
    y_path = f"{OUT}/xgb_valid_y.npy"
    if not (os.path.exists(proba_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"{proba_path} 또는 {y_path}가 없습니다. 14_임계값_최적화.py를 먼저 실행해주세요."
        )
    p_val = np.load(proba_path)
    y_val = np.load(y_path)
    cand_thresholds = np.linspace(0.01, 0.99, 981)

    costs = []
    for th in cand_thresholds:
        pred = (p_val >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        costs.append(fn * ratio + fp * 1)
    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    th_best = cand_thresholds[best_idx]

    pred = (p_val >= th_best).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    result = pd.DataFrame([{
        "배율": round(ratio, 2), "임계값": round(th_best, 4),
        "Accuracy": round(accuracy_score(y_val, pred), 4),
        "Precision": round(precision_score(y_val, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_val, pred, zero_division=0), 4),
        "F1": round(f1_score(y_val, pred, zero_division=0), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }])
    result.to_csv(f"{OUT}/LGD순이익률_기반_임계값_결과.csv", index=False, encoding=ENC)
    log(f"최종 임계값 = {th_best:.4f}")
    print(result.to_string(index=False))
    log("완료: LGD순이익률_배율_계산결과.csv, LGD순이익률_기반_임계값_결과.csv 저장됨")
