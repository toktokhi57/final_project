# -*- coding: utf-8 -*-
"""
STEP 13-1. 임계값 하나씩 직접 테스트
- 목적: 14번은 여러 임계값/비율을 자동으로 한 번에 훑어보는 스크립트였다면, 이건 원하는
  임계값 하나를 직접 입력해서 그 지점의 성능(Accuracy/Precision/Recall/F1/혼동행렬)을
  바로 확인해보는 용도다.
- 14번이 저장해둔 XGBoost의 valid set 예측 확률(xgb_valid_proba.npy)을 그대로 불러오기
  때문에 XGBoost를 매번 재학습하지 않는다 - 수 초 안에 결과가 나온다.
- 실행할 때마다 결과가 임계값_직접테스트_기록.csv에 한 줄씩 누적 저장되므로, 여러 값을
  이것저것 테스트해본 뒤 그 기록을 모아서 최종 비교표로 써도 된다.

사용법: 아래 THRESHOLD 값만 원하는 숫자(0~1 사이)로 바꿔서 실행하면 된다.

주의: 이 스크립트를 실행하기 전에 14번(임계값_최적화.py)을 먼저 한 번 실행해서
  xgb_valid_proba.npy / xgb_valid_y.npy 파일이 만들어져 있어야 한다. 08번 인코딩이나
  XGBoost 모델 자체를 바꿨다면 14번을 다시 실행해서 이 파일들을 새로 갱신해야 한다.

입력: xgb_valid_proba.npy, xgb_valid_y.npy (14번 출력)
출력: 임계값_직접테스트_기록.csv (실행할 때마다 한 줄씩 누적, 덮어쓰지 않음)
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
)

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"

# ============================================================
# 여기 임계값만 원하는 값으로 바꿔서 실행하면 됨 (0~1 사이)
# 0.665 = 최종 확정값 (FN:FP 배율 4:1 대표값, 이자계산/LGD+순이익률/외부금리기준 등
# 여러 방식이 3~5:1로 수렴한 결과를 종합해 채택. 상세 근거는 15/15_1번 참고)
THRESHOLD = 0.665
# ============================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


if __name__ == "__main__":
    proba_path = f"{OUT}/xgb_valid_proba.npy"
    y_path = f"{OUT}/xgb_valid_y.npy"
    if not (os.path.exists(proba_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"{proba_path} 또는 {y_path}가 없습니다. "
            "14_임계값_최적화.py를 먼저 한 번 실행해서 이 파일들을 만들어주세요."
        )

    p_val = np.load(proba_path)
    y_val = np.load(y_path)

    pred = (p_val >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    acc = accuracy_score(y_val, pred)
    prec = precision_score(y_val, pred, zero_division=0)
    rec = recall_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)
    auc = roc_auc_score(y_val, p_val)  # 임계값과 무관 (참고용)

    log("=" * 60)
    log(f"임계값 = {THRESHOLD}")
    log(f"AUC(참고, 임계값과 무관) = {auc:.4f}")
    log(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    log(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    log(f"부도자 중 놓친 비율(FN/실제부도) = {fn / (fn + tp):.4f}  |  "
        f"정상고객 중 오거절 비율(FP/실제정상) = {fp / (fp + tn):.4f}")

    row = {
        "실행시각": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Threshold": THRESHOLD, "AUC": round(auc, 4),
        "Accuracy": round(acc, 4), "Precision": round(prec, 4),
        "Recall": round(rec, 4), "F1": round(f1, 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    log_path = f"{OUT}/임계값_직접테스트_기록.csv"
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(log_path)
    df_row.to_csv(log_path, mode="a", header=write_header, index=False, encoding=ENC)
    log(f"결과를 {log_path}에 누적 저장했습니다 (기존 기록은 유지되고 이번 결과가 새 줄로 추가됨).")
