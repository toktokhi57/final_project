# -*- coding: utf-8 -*-
"""
STEP 21. 최종 테스트셋 검증 (valid셋 재사용 문제 해소)

배경 (왜 필요한가):
- 지금까지 임계값 확정(14/15번), 확률보정 검증(17번), 등급구간 확정(18번), 등급 프로파일링
  (19번), 공정성 검증(20번)까지 전부 "같은 valid셋"(61,501행) 하나로 판단해왔다.
  같은 데이터를 계속 재사용해서 여러 결정을 내리면, 그 결정들이 이 valid셋 하나에만
  우연히 잘 맞는(=과최적화된) 것일 수 있다는 위험이 있는데, 지금까지는 이걸 검증할
  방법이 없었다.
- 이 스크립트는 지금까지 "확정된" 규칙들(임계값=0.665, Platt 보정, 등급구간=기저율중심
  방식의 고정 확률경계)을 그대로 가져와서, 지금까지 단 한 번도 어떤 결정에도 쓰이지
  않은 완전히 새로운 test셋에 적용해보고, 결과가 valid셋에서 봤던 것과 비슷하게
  나오는지 확인한다. 비슷하면 지금까지의 결정이 valid셋에 과최적화된 게 아니라 실제로
  일반화된다는 뜻이고, 크게 다르면 재검토가 필요하다는 신호다.

절차:
  1) 기존 train+valid(총 약 30.8만행)를 다시 하나로 합친 뒤, 70/15/15 비율로
     new_train(모델 학습 전용) / new_valid(보정기 학습 전용) / new_test(최종 검증 전용 -
     지금부터 다른 어떤 목적에도 쓰지 않음)로 새로 나눈다.
  2) new_train으로 XGBoost를 12번과 동일 파라미터로 재학습
  3) new_valid로 Platt 보정기를 새로 학습 (17번과 동일 방식)
  4) new_test에서 딱 한 번만 평가:
     - 임계값 0.665 기준 Accuracy/Precision/Recall/F1 (14_1번 결과와 비교)
     - 보정 전후 Brier score (17번 결과와 비교)
     - 18번에서 확정한 고정 확률경계(A<2.02%, B<4.04%, C<8.07%, D<16.15%, E>=16.15%)로
       등급을 매겨서 등급별 실제부도율이 여전히 단조증가하는지 확인 (18번 결과와 비교)
  5) "원래 valid셋 기준 결과" vs "새 test셋 기준 결과"를 나란히 표로 비교

주의: new_train은 원래 train(약 24.6만행)보다 조금 작다(70% 기준 약 21.5만행). 그래서
      AUC가 원래보다 아주 약간 낮게 나올 수 있는데, 이건 "재사용 문제"가 아니라 순수하게
      학습 데이터가 줄어든 효과이니 감안해서 봐야 한다.

입력: tree_encoded_train.pkl, tree_encoded_valid.pkl (08번 출력)
출력: 최종테스트셋_검증_결과.csv, 최종테스트셋_등급별검증.csv
"""
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss, accuracy_score,
                              precision_score, recall_score, f1_score, confusion_matrix)

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42
FINAL_THRESHOLD = 0.665
GRADES = ["A", "B", "C", "D", "E"]

# 18번에서 확정한 고정 확률경계 (원본 valid 기저율 8.07304%를 기준으로 이미 확정된
# 절대값 - 새 test셋에서도 재계산하지 않고 이 고정값을 그대로 적용한다. 실전에서도 매
# 배치마다 기준을 새로 바꾸지 않고 한 번 정한 기준을 계속 쓰는 것과 동일한 방식이다)
ORIGINAL_BASE_RATE = 0.08073039462772963
GRADE_BINS = [-0.001, ORIGINAL_BASE_RATE / 4, ORIGINAL_BASE_RATE / 2,
              ORIGINAL_BASE_RATE, ORIGINAL_BASE_RATE * 2, 1.001]

# 기존(원래 valid셋 기준) 확정 결과 - 비교용 참고치
REF_AUC = 0.7910  # 20번 모델A(전체변수 포함) 기준
REF_ACC, REF_PREC, REF_REC, REF_F1 = 0.8641, 0.2816, 0.4407, 0.3436  # 14_1번
REF_BRIER_RAW, REF_BRIER_PLATT = 0.16660577058792114, 0.06588560342788696  # 17번
REF_GRADE_RATES = {"A": 1.3042, "B": 2.6374, "C": 5.8321, "D": 11.2310, "E": 26.9176}  # 18번(방법3)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


if __name__ == "__main__":
    t0 = time.time()

    # =====================================================================
    # 1) train+valid 재통합 후 70/15/15로 재분할
    # =====================================================================
    log("=" * 70)
    log("기존 train+valid 재통합 -> 70(new_train)/15(new_valid)/15(new_test) 재분할")
    tr = pd.read_pickle(f"{OUT}/tree_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl")
    pool = pd.concat([tr, val], axis=0, ignore_index=True)
    del tr, val

    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols = [c for c in pool.columns if c not in exclude]
    med = pool[feat_cols].median(numeric_only=True)
    X_all = pool[feat_cols].fillna(med)
    y_all = pool[TARGET].astype(int).values

    # 1차: test(15%) 분리
    X_rest, X_test, y_rest, y_test = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_STATE
    )
    # 2차: 나머지(85%)를 train(약 82.35%)/valid(약 17.65%)로 분리 -> 전체 기준 70/15
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_rest, y_rest, test_size=0.15 / 0.85, stratify=y_rest, random_state=RANDOM_STATE
    )
    log(f"new_train={X_train.shape}, new_valid={X_valid.shape}, new_test={X_test.shape} "
        f"(new_test는 지금부터 이 스크립트의 마지막 평가에만 사용)")

    # =====================================================================
    # 2) new_train으로 XGBoost 재학습 (12번과 동일 파라미터)
    # =====================================================================
    log("=" * 70)
    log("XGBoost 재학습 중 (new_train)...")
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    spw = n_neg / n_pos
    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="auc",
                         random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")
    xgb.fit(X_train, y_train)
    log("학습 완료")

    p_valid_raw = xgb.predict_proba(X_valid)[:, 1]
    p_test_raw = xgb.predict_proba(X_test)[:, 1]

    # =====================================================================
    # 3) new_valid로 Platt 보정기 학습 (17번과 동일 방식)
    # =====================================================================
    log("=" * 70)
    log("Platt 보정기 학습 중 (new_valid)...")
    eps = 1e-6

    def to_logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    platt = LogisticRegression()
    platt.fit(to_logit(p_valid_raw), y_valid)
    p_test_platt = platt.predict_proba(to_logit(p_test_raw))[:, 1]

    # =====================================================================
    # 4) new_test에서 딱 한 번 최종 평가
    # =====================================================================
    log("=" * 70)
    log(">>> new_test 최종 평가 (지금까지 어떤 결정에도 사용된 적 없는 데이터) <<<")

    auc_test = roc_auc_score(y_test, p_test_raw)
    pred_test = (p_test_raw >= FINAL_THRESHOLD).astype(int)
    acc = accuracy_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test, zero_division=0)
    rec = recall_score(y_test, pred_test, zero_division=0)
    f1 = f1_score(y_test, pred_test, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()

    brier_raw = brier_score_loss(y_test, p_test_raw)
    brier_platt = brier_score_loss(y_test, p_test_platt)

    log(f"[new_test] AUC={auc_test:.4f}  (참고: 원래 valid 기준 {REF_AUC:.4f})")
    log(f"[new_test] 임계값={FINAL_THRESHOLD} 기준 -> Acc={acc:.4f}, Prec={prec:.4f}, "
        f"Rec={rec:.4f}, F1={f1:.4f}  (TN={tn}, FP={fp}, FN={fn}, TP={tp})")
    log(f"  참고(원래 valid): Acc={REF_ACC:.4f}, Prec={REF_PREC:.4f}, Rec={REF_REC:.4f}, F1={REF_F1:.4f}")
    log(f"[new_test] Brier score: 보정전={brier_raw:.4f}, Platt보정후={brier_platt:.4f}")
    log(f"  참고(원래 valid): 보정전={REF_BRIER_RAW:.4f}, Platt보정후={REF_BRIER_PLATT:.4f}")

    compare_df = pd.DataFrame([
        {"지표": "AUC", "원래valid": REF_AUC, "새new_test": auc_test, "차이": auc_test - REF_AUC},
        {"지표": "Accuracy", "원래valid": REF_ACC, "새new_test": acc, "차이": acc - REF_ACC},
        {"지표": "Precision", "원래valid": REF_PREC, "새new_test": prec, "차이": prec - REF_PREC},
        {"지표": "Recall", "원래valid": REF_REC, "새new_test": rec, "차이": rec - REF_REC},
        {"지표": "F1", "원래valid": REF_F1, "새new_test": f1, "차이": f1 - REF_F1},
        {"지표": "Brier(보정전)", "원래valid": REF_BRIER_RAW, "새new_test": brier_raw,
         "차이": brier_raw - REF_BRIER_RAW},
        {"지표": "Brier(Platt후)", "원래valid": REF_BRIER_PLATT, "새new_test": brier_platt,
         "차이": brier_platt - REF_BRIER_PLATT},
    ])
    compare_df.to_csv(f"{OUT}/최종테스트셋_검증_결과.csv", index=False, encoding=ENC)
    print(compare_df.round(4).to_string(index=False))

    # =====================================================================
    # 5) 18번 고정 등급구간을 new_test에 적용 -> 단조성 재확인
    # =====================================================================
    log("=" * 70)
    log("18번 고정 등급구간(기저율중심)을 new_test에 적용해 재검증 중...")
    grade_test = pd.cut(p_test_platt, bins=GRADE_BINS, labels=GRADES)
    grade_df = pd.DataFrame({"등급": grade_test, "실제": y_test})
    grade_summary = (
        grade_df.groupby("등급", observed=False)
        .agg(인원수=("실제", "count"), 부도자수=("실제", "sum"))
        .reindex(GRADES)
    )
    grade_summary["실제부도율(%)"] = grade_summary["부도자수"] / grade_summary["인원수"] * 100
    grade_summary["참고_원래valid부도율(%)"] = [REF_GRADE_RATES[g] for g in GRADES]

    rates = grade_summary["실제부도율(%)"].values
    is_monotonic = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    log(f"new_test에서도 등급별 실제부도율 단조증가 여부: "
        f"{'예 (기존 결정이 새 데이터에서도 일반화됨)' if is_monotonic else '아니오 (역전 발생 - 재검토 필요)'}")
    print(grade_summary.round(4).to_string())
    grade_summary.to_csv(f"{OUT}/최종테스트셋_등급별검증.csv", encoding=ENC)

    log("=" * 70)
    log("저장 완료: 최종테스트셋_검증_결과.csv, 최종테스트셋_등급별검증.csv")
    log(f"전체 소요시간: {time.time() - t0:.1f}초")
