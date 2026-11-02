# -*- coding: utf-8 -*-
"""
STEP 9. 기준 모델(Baseline) 구축
- 목적: 이후 만든/만들 모든 모델(RF/XGB/LGBM/WoE-로지스틱 등)이 "이 정도는 넘어야 의미가
  있다"는 절대 하한선과, 별다른 튜닝 없이 표준적으로 시도해볼 수 있는 해석 가능한 기준선을
  동시에 확보한다.

  - DummyClassifier : 데이터를 실제로 보지 않고 정답의 분포만으로 찍는 모델.
    아래 지표보다 낮으면 그 모델은 "데이터에서 아무것도 배우지 못했다"는 뜻이므로
    폐기해야 한다. 3가지 전략을 모두 본다:
      * most_frequent : 무조건 다수 클래스(정상)만 예측.
        Accuracy는 부도율(약 8%)의 반대인 92%대로 높게 나오지만 AUC는 정확히 0.5,
        Recall은 0 -> "정확도만 보면 안 되는 이유"를 숫자로 그대로 보여줌
      * stratified     : 학습 데이터의 클래스 비율(정상 92% / 부도 8%)대로 무작위로 찍음.
        이게 진짜 의미의 "무작위 추측 점수" -> AUC는 이론상 0.5 근처, 이 밑으로 모델이
        떨어지면 데이터를 아예 안 보고 확률만 흉내내는 것보다도 못하다는 뜻
      * uniform        : 완전 50:50 동전던지기
  - LogisticRegression(표준 베이스라인) : class_weight 등 불균형 보정 없이, 08번에서 만든
    원-핫+표준화 피처에 그대로 적용한 가장 기본적인 로지스틱 회귀. "기본기만으로는 어디까지
    되는지"를 보여주는 해석 가능한 기준선. 이 모델은 Recall이 매우 낮게 나오는데(부도자를
    거의 못 잡음), 이건 버그가 아니라 의도된 결과다 - 불균형 데이터에서 보정 없이 학습하면
    다수 클래스(정상)쪽으로 예측이 쏠리는 전형적인 현상을 그대로 보여주는 것이 이 기준선의
    목적이다.
  - LogisticRegression(class_weight=balanced) : 위 표준 베이스라인과 똑같은 조건에서
    class_weight="balanced"만 추가한 버전. 표준 베이스라인과 나란히 비교하면 "불균형
    보정 하나만 켰을 때 Recall이 실제로 얼마나 뛰는지"를 이 파일 안에서 바로 확인할 수 있다
    (08번에서 이미 확인한 것과 같은 효과를 여기서도 재확인하는 용도).

- 입력: 08_인코딩_스케일링.py 가 만든 linear_encoded_train/valid.pkl
  (DummyClassifier는 X를 실제로 쓰진 않지만, sklearn API 형식을 맞추기 위해 같은 X를 재사용)
- 출력: 모델비교_기준모델_Dummy_로지스틱.csv
"""
import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
)

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def evaluate(y_true, pred, threshold=0.5):
    pred_label = (pred >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, pred),
        "Accuracy": accuracy_score(y_true, pred_label),
        "Precision": precision_score(y_true, pred_label, zero_division=0),
        "Recall": recall_score(y_true, pred_label, zero_division=0),
        "F1": f1_score(y_true, pred_label, zero_division=0),
    }


if __name__ == "__main__":
    t0 = time.time()

    tr = pd.read_pickle(f"{OUT}/linear_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/linear_encoded_valid.pkl")
    feat_cols = [c for c in tr.columns if c != TARGET]
    X_tr, X_val = tr[feat_cols].values, val[feat_cols].values
    y_tr, y_val = tr[TARGET].astype(int).values, val[TARGET].astype(int).values
    log(f"로드 완료: train {X_tr.shape}, valid {X_val.shape}, "
        f"부도율 train={y_tr.mean():.4f} / valid={y_val.mean():.4f}")

    results = {}

    # ---------------------------------------------------------------------
    # 1) DummyClassifier 3종 - 절대 하한선
    # ---------------------------------------------------------------------
    log("=" * 70)
    log("DummyClassifier 3종 학습 (절대 하한선)")
    for strategy, name in [
        ("most_frequent", "Dummy-최빈클래스"),
        ("stratified", "Dummy-비율기반무작위"),
        ("uniform", "Dummy-완전무작위(50:50)"),
    ]:
        dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
        dummy.fit(X_tr, y_tr)
        p_tr = dummy.predict_proba(X_tr)[:, 1]
        p_val = dummy.predict_proba(X_val)[:, 1]
        r_tr, r_val = evaluate(y_tr, p_tr), evaluate(y_val, p_val)
        results[name] = {"Train": r_tr, "Valid": r_val}
        log(f"[{name}] Valid AUC={r_val['AUC']:.4f}, Accuracy={r_val['Accuracy']:.4f}, "
            f"Recall={r_val['Recall']:.4f}")

    # ---------------------------------------------------------------------
    # 2) LogisticRegression - 해석 가능한 표준 베이스라인 (불균형 보정 없음)
    # ---------------------------------------------------------------------
    log("=" * 70)
    log("LogisticRegression 학습 (표준 베이스라인, class_weight 없음)")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    p_tr, p_val = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_val)[:, 1]
    r_tr, r_val = evaluate(y_tr, p_tr), evaluate(y_val, p_val)
    results["로지스틱회귀(표준 베이스라인)"] = {"Train": r_tr, "Valid": r_val}
    log(f"[로지스틱회귀(표준 베이스라인)] Valid AUC={r_val['AUC']:.4f}, "
        f"Accuracy={r_val['Accuracy']:.4f}, Recall={r_val['Recall']:.4f}")

    # ---------------------------------------------------------------------
    # 3) LogisticRegression - class_weight="balanced" 버전 (불균형 보정 효과 비교용)
    # ---------------------------------------------------------------------
    log("=" * 70)
    log("LogisticRegression 학습 (class_weight='balanced' 추가)")
    model_bal = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                    random_state=RANDOM_STATE)
    model_bal.fit(X_tr, y_tr)
    p_tr_bal, p_val_bal = model_bal.predict_proba(X_tr)[:, 1], model_bal.predict_proba(X_val)[:, 1]
    r_tr_bal, r_val_bal = evaluate(y_tr, p_tr_bal), evaluate(y_val, p_val_bal)
    results["로지스틱회귀(class_weight=balanced)"] = {"Train": r_tr_bal, "Valid": r_val_bal}
    log(f"[로지스틱회귀(class_weight=balanced)] Valid AUC={r_val_bal['AUC']:.4f}, "
        f"Accuracy={r_val_bal['Accuracy']:.4f}, Recall={r_val_bal['Recall']:.4f}")
    log(f"-> 표준 베이스라인 대비 Recall 변화: {r_val['Recall']:.4f} -> {r_val_bal['Recall']:.4f} "
        f"(AUC는 {r_val['AUC']:.4f} -> {r_val_bal['AUC']:.4f}로 거의 그대로 - "
        f"판별력 자체가 아니라 판정 임계값 쏠림만 교정된 것임을 보여줌)")

    # ---------------------------------------------------------------------
    # 결과 정리
    # ---------------------------------------------------------------------
    rows = []
    for name, r in results.items():
        row = {"모델": name}
        for split in ("Train", "Valid"):
            for metric, v in r[split].items():
                row[f"{split}_{metric}"] = round(v, 4)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(f"{OUT}/모델비교_기준모델_Dummy_로지스틱.csv", index=False, encoding=ENC)

    log("=" * 70)
    log("=== 기준 모델 비교 결과 ===")
    print(comparison.to_string(index=False))
    log("-> 이후 만드는 모든 모델(RF/XGB/LGBM/WoE-로지스틱 등)은 최소한 "
        "'Dummy-비율기반무작위'의 AUC(0.5 근처)보다는 확실히 높아야 하고, "
        "'로지스틱회귀(표준 베이스라인)'보다는 나아야 그 복잡도를 들인 의미가 있다.")
    log(f"전체 소요시간: {time.time()-t0:.1f}초")
