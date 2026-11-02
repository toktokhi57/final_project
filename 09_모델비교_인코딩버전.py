# -*- coding: utf-8 -*-
"""
STEP 8. 모델 비교 (08번 인코딩 버전) - RandomForest/XGBoost/LightGBM/로지스틱(원핫+스케일)/WoE-로지스틱
- 목적: 05/05b/05c번은 각 스크립트 안에서 대충 인코딩(트리=라벨인코딩, 로지스틱=범주형 아예
  제외하고 수치형만 사용)해서 비교한 거라 로지스틱 쪽이 불리한 비교였다. 08번에서 모델 계열별로
  제대로 인코딩·스케일링한 파일을 각 모델에 맞게 넣어서 공정하게 다시 비교한다.
  - 트리 3종            : tree_encoded_train/valid.pkl (라벨인코딩)
  - 로지스틱(일반)       : linear_encoded_train/valid.pkl (원-핫 + StandardScaler)
  - 로지스틱(WoE 스코어카드) : woe_encoded_train/valid.pkl (Weight of Evidence 치환)
  모두 08번에서 같은 seed(RANDOM_STATE=42)로 나눈 동일한 train/valid 사람들이라 공정 비교 가능.
- 입력: 08_인코딩_스케일링.py 가 만든 6개 pkl 파일
- 출력: 모델비교_LR_RF_XGB_LGBM_WoE_인코딩버전.csv, 모델비교_ROC_curve_인코딩버전.png,
        변수중요도_비교_top15_인코딩버전.csv (트리 3종 기준),
        WoE_로지스틱_계수_top15.csv (WoE 스코어카드 해석용 계수)
"""
import time
import gc
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

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


def plot_roc(curves: dict, path):
    plt.figure(figsize=(7, 7))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("모델별 ROC Curve 비교 - 08번 인코딩 버전 (Valid set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 트리 3종 (RF/XGB/LGBM) : tree_encoded 사용, n_jobs=1 고정
#   (macOS에서 n_jobs=-1이 TimeoutError: [Errno 60]을 유발하는 문제 회피)
# ---------------------------------------------------------------------------
def run_tree_models(results, roc_curves):
    log("=" * 70)
    log("트리 계열 (tree_encoded_*.pkl) 로드")
    tr = pd.read_pickle(f"{OUT}/tree_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl")
    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols = [c for c in tr.columns if c not in exclude]
    X_tr, X_val = tr[feat_cols], val[feat_cols]
    y_tr, y_val = tr[TARGET].astype(int).values, val[TARGET].astype(int).values
    log(f"트리용 피처 {len(feat_cols)}개, train {X_tr.shape}, valid {X_val.shape}")

    med = X_tr.median(numeric_only=True)
    X_tr_rf, X_val_rf = X_tr.fillna(med), X_val.fillna(med)  # RandomForest는 결측 허용 안 함

    log("RandomForest 학습 중...")
    t1 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=50,
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE,
    )
    rf.fit(X_tr_rf, y_tr)
    p_tr, p_val = rf.predict_proba(X_tr_rf)[:, 1], rf.predict_proba(X_val_rf)[:, 1]
    results["RandomForest"] = {"Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val),
                                "학습시간(초)": round(time.time() - t1, 1)}
    roc_curves["RandomForest"] = (y_val, p_val)

    log("XGBoost 학습 중...")
    t1 = time.time()
    n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    xgb = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=n_neg / n_pos,
        eval_metric="auc", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist",
    )
    xgb.fit(X_tr, y_tr)
    p_tr, p_val = xgb.predict_proba(X_tr)[:, 1], xgb.predict_proba(X_val)[:, 1]
    results["XGBoost"] = {"Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val),
                           "학습시간(초)": round(time.time() - t1, 1)}
    roc_curves["XGBoost"] = (y_val, p_val)

    log("LightGBM 학습 중...")
    t1 = time.time()
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=n_neg / n_pos,
        random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
    )
    lgbm.fit(X_tr, y_tr)
    p_tr, p_val = lgbm.predict_proba(X_tr)[:, 1], lgbm.predict_proba(X_val)[:, 1]
    results["LightGBM"] = {"Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val),
                            "학습시간(초)": round(time.time() - t1, 1)}
    roc_curves["LightGBM"] = (y_val, p_val)

    imp_df = pd.DataFrame({
        "컬럼명": feat_cols,
        "RandomForest_중요도": rf.feature_importances_,
        "XGBoost_중요도": xgb.feature_importances_,
        "LightGBM_중요도": lgbm.feature_importances_,
    })
    for c in ["RandomForest_중요도", "XGBoost_중요도", "LightGBM_중요도"]:
        imp_df[c] = imp_df[c] / imp_df[c].sum()
    imp_df["평균순위점수"] = imp_df[["RandomForest_중요도", "XGBoost_중요도", "LightGBM_중요도"]].mean(axis=1)
    imp_df = imp_df.sort_values("평균순위점수", ascending=False).reset_index(drop=True)
    imp_df.head(15).to_csv(f"{OUT}/변수중요도_비교_top15_인코딩버전.csv", index=False, encoding=ENC)

    del tr, val, X_tr, X_val, X_tr_rf, X_val_rf
    gc.collect()


# ---------------------------------------------------------------------------
# 로지스틱(일반) : linear_encoded 사용 (원-핫 + StandardScaler)
# ---------------------------------------------------------------------------
def run_linear_logistic(results, roc_curves):
    log("=" * 70)
    log("로지스틱(일반) (linear_encoded_*.pkl) 로드")
    tr = pd.read_pickle(f"{OUT}/linear_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/linear_encoded_valid.pkl")
    feat_cols = [c for c in tr.columns if c != TARGET]
    X_tr, X_val = tr[feat_cols].values, val[feat_cols].values
    y_tr, y_val = tr[TARGET].astype(int).values, val[TARGET].astype(int).values
    log(f"로지스틱(일반)용 피처 {len(feat_cols)}개, train {X_tr.shape}, valid {X_val.shape}")

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    p_tr, p_val = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_val)[:, 1]
    results["로지스틱(원핫+표준화)"] = {"Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val)}
    roc_curves["로지스틱(원핫+표준화)"] = (y_val, p_val)

    del tr, val, X_tr, X_val
    gc.collect()


# ---------------------------------------------------------------------------
# WoE-로지스틱 (신용평가 스코어카드) : woe_encoded 사용
# ---------------------------------------------------------------------------
def run_woe_logistic(results, roc_curves):
    log("=" * 70)
    log("WoE-로지스틱(신용평가 스코어카드) (woe_encoded_*.pkl) 로드")
    tr = pd.read_pickle(f"{OUT}/woe_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/woe_encoded_valid.pkl")
    feat_cols = [c for c in tr.columns if c != TARGET]
    X_tr, X_val = tr[feat_cols].values, val[feat_cols].values
    y_tr, y_val = tr[TARGET].astype(int).values, val[TARGET].astype(int).values
    log(f"WoE 피처 {len(feat_cols)}개, train {X_tr.shape}, valid {X_val.shape}")

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    p_tr, p_val = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_val)[:, 1]
    results["WoE-로지스틱(스코어카드)"] = {"Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val)}
    roc_curves["WoE-로지스틱(스코어카드)"] = (y_val, p_val)

    # WoE 계수는 그 자체로 "이 변수가 위험도에 얼마나 기여하는지"를 바로 해석할 수 있어서
    # 스코어카드 문서화에 그대로 쓸 수 있음 (계수가 클수록 해당 변수의 영향력이 큼)
    coef_df = pd.DataFrame({"컬럼명": feat_cols, "계수": model.coef_[0]})
    coef_df["절대값"] = coef_df["계수"].abs()
    coef_df = coef_df.sort_values("절대값", ascending=False).drop(columns="절대값").reset_index(drop=True)
    coef_df.head(15).to_csv(f"{OUT}/WoE_로지스틱_계수_top15.csv", index=False, encoding=ENC)
    log("WoE 로지스틱 계수 상위 15개 저장 완료")

    del tr, val, X_tr, X_val
    gc.collect()


if __name__ == "__main__":
    t0 = time.time()
    results, roc_curves = {}, {}

    run_tree_models(results, roc_curves)
    run_linear_logistic(results, roc_curves)
    run_woe_logistic(results, roc_curves)

    rows = []
    for name, r in results.items():
        row = {"모델": name}
        for split in ("Train", "Valid"):
            for metric, v in r[split].items():
                row[f"{split}_{metric}"] = round(v, 4)
        row["학습시간(초)"] = r.get("학습시간(초)", "-")
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(f"{OUT}/모델비교_LR_RF_XGB_LGBM_WoE_인코딩버전.csv", index=False, encoding=ENC)
    log("=" * 70)
    log("=== 모델 비교 결과 (08번 인코딩 버전) ===")
    print(comparison.to_string(index=False))

    plot_roc(roc_curves, f"{OUT}/모델비교_ROC_curve_인코딩버전.png")
    log(f"전체 소요시간: {time.time()-t0:.1f}초")
