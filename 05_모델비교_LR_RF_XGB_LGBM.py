# -*- coding: utf-8 -*-
"""
STEP 5-보조. 모델 비교: WoE+로지스틱회귀(스코어카드) vs RandomForest vs XGBoost vs LightGBM
- 목적: "해석력 vs 예측성능" 트레이드오프를 같은 데이터·같은 train/valid 분할로 직접 확인
- 입력: 04_결측치_이상치_분석.py 가 만든 full_feature_table_imputed.pkl
- 출력: 모델별 AUC/정확도/정밀도/재현율 비교표(csv), ROC curve 비교 그림(png),
        변수중요도 비교표(csv)

주의: RandomForest/XGBoost/LightGBM은 scikit-learn, xgboost, lightgbm 패키지가 필요합니다.
      본인 컴퓨터에 설치되어 있어야 실행됩니다.
      (pip install xgboost lightgbm  ->  scikit-learn은 이미 BigMart 코드에서 쓰셨으니 설치돼 있을 겁니다)
"""
import time
import gc
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

UP = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋"  # iCloud 동기화 문제로 Desktop 밖으로 이동
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# 1) 데이터 로드 + 04단계와 동일한 방식의 stratified 80/20 분할
#    (모든 모델이 완전히 같은 사람들로 학습/검증하도록 맞춰야 공정한 비교가 됨)
# ---------------------------------------------------------------------------
def stratified_split(train, valid_ratio=0.2, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    idx0 = train.index[train[TARGET] == 0].to_numpy().copy()
    idx1 = train.index[train[TARGET] == 1].to_numpy().copy()
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0, n1 = int(len(idx0) * valid_ratio), int(len(idx1) * valid_ratio)
    val_idx = np.concatenate([idx0[:n0], idx1[:n1]])
    tr_idx = np.concatenate([idx0[n0:], idx1[n1:]])
    return train.loc[tr_idx].reset_index(drop=True), train.loc[val_idx].reset_index(drop=True)


def load_train_valid():
    df = pd.read_pickle(f"{OUT}/full_feature_table_imputed.pkl")
    train = df[df["IS_TRAIN"] == 1].copy()
    del df  # test 행까지 포함된 원본은 더 필요 없으므로 즉시 해제
    gc.collect()
    train[TARGET] = train[TARGET].astype(int)
    log(f"학습대상(train) 전체: {train.shape}")
    tr, val = stratified_split(train)
    del train
    gc.collect()
    log(f"분할 완료 -> train: {tr.shape}, valid: {val.shape}, "
        f"부도율 train={tr[TARGET].mean():.4f} / valid={val[TARGET].mean():.4f}")
    return tr, val


# ---------------------------------------------------------------------------
# 2) 트리 계열(RF/XGB/LGBM) 공용 피처 행렬 준비
#    - 범주형은 Label Encoding (트리 모델은 원-핫 없이도 분기로 잘 처리함)
#    - 결측치: XGBoost/LightGBM은 결측을 그대로 넣으면 알아서 최적 분기방향을 학습함(네이티브 지원).
#              RandomForest(sklearn)는 결측 허용이 안 되므로 중앙값으로 별도 대치.
# ---------------------------------------------------------------------------
def build_tree_features(tr, val):
    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols = [c for c in tr.columns if c not in exclude]

    X_tr, X_val = tr[feat_cols].copy(), val[feat_cols].copy()
    cat_cols = X_tr.select_dtypes(include="object").columns.tolist()

    for c in cat_cols:
        cats = pd.Categorical(pd.concat([X_tr[c], X_val[c]], axis=0).fillna("MISSING"))
        mapping = {cat: i for i, cat in enumerate(cats.categories)}
        X_tr[c] = X_tr[c].fillna("MISSING").map(mapping).astype(int)
        X_val[c] = X_val[c].fillna("MISSING").map(mapping).astype(int)

    log(f"트리모델용 피처 {len(feat_cols)}개 준비 완료 (범주형 {len(cat_cols)}개 라벨인코딩)")
    return X_tr, X_val, feat_cols


def median_impute_for_rf(X_tr, X_val):
    """RandomForest 전용: 남은 결측을 train 기준 중앙값으로 채움."""
    med = X_tr.median(numeric_only=True)
    return X_tr.fillna(med), X_val.fillna(med)


# ---------------------------------------------------------------------------
# 3) 로지스틱 회귀
#    - 참고: 기존 home_credit_credit_grading_pipeline.py의 WoE+스코어카드 결과(Valid AUC 0.755)를
#      이 비교표의 "로지스틱회귀(WoE 스코어카드)" 행에 그대로 함께 적어서 비교하는 걸 추천합니다.
#      아래 함수는 이 파일 하나만으로 돌아가도록 만든 "표준화(StandardScaler) 버전"입니다.
#      WoE 인코딩 없이 순수 수치 표준화만 하면 로지스틱회귀 성능이 어떻게 나오는지 보여주는
#      용도로, RF/XGB/LGBM과 동일한 raw 피처·동일한 분할을 사용해 공정하게 비교합니다.
# ---------------------------------------------------------------------------
def train_simple_logistic(X_tr_raw, X_val_raw, y_tr, y_val):
    numeric_cols = X_tr_raw.select_dtypes(include="number").columns.tolist()
    X_tr_num = X_tr_raw[numeric_cols].fillna(X_tr_raw[numeric_cols].median())
    X_val_num = X_val_raw[numeric_cols].fillna(X_tr_raw[numeric_cols].median())

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_num)
    X_val_s = scaler.transform(X_val_num)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    model.fit(X_tr_s, y_tr)
    return model.predict_proba(X_tr_s)[:, 1], model.predict_proba(X_val_s)[:, 1]


# ---------------------------------------------------------------------------
# 4) RandomForest / XGBoost / LightGBM 학습
#    n_jobs=1로 고정 (병렬처리 끔): n_jobs=-1(전체 코어 사용)로 두면 macOS에서
#    방화벽/보안소프트웨어/VPN이 걸려있을 때 병렬 워커끼리 통신용 로컬 소켓을 여는
#    과정에서 TimeoutError: [Errno 60] Operation timed out 이 나는 경우가 흔하다.
#    n_jobs=1은 코어 하나만 쓰므로 조금 느리지만 이런 문제 없이 안정적으로 돌아간다.
#    환경 문제가 해결되면 n_jobs=4 정도로 다시 올려도 된다.
# ---------------------------------------------------------------------------
def train_random_forest(X_tr, X_val, y_tr):
    X_tr_i, X_val_i = median_impute_for_rf(X_tr, X_val)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=50,
        class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE,
    )
    model.fit(X_tr_i, y_tr)
    return model, model.predict_proba(X_tr_i)[:, 1], model.predict_proba(X_val_i)[:, 1]


def train_xgboost(X_tr, X_val, y_tr):
    n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,  # 불균형 데이터 보정 (부도=소수 클래스)
        eval_metric="auc", random_state=RANDOM_STATE, n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_tr)[:, 1], model.predict_proba(X_val)[:, 1]


def train_lightgbm(X_tr, X_val, y_tr):
    n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    model = LGBMClassifier(
        n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,
        random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_tr)[:, 1], model.predict_proba(X_val)[:, 1]


# ---------------------------------------------------------------------------
# 5) 공통 평가
# ---------------------------------------------------------------------------
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
    plt.title("모델별 ROC Curve 비교 (Valid set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    t0 = time.time()
    tr, val = load_train_valid()
    y_tr, y_val = tr[TARGET].values, val[TARGET].values

    X_tr_raw, X_val_raw, feat_cols = build_tree_features(tr, val)
    del tr, val  # y_tr/y_val은 이미 뽑아뒀고, tr/val 원본은 이 시점부턴 안 쓰이므로 해제
    gc.collect()

    results = {}
    roc_curves = {}

    # --- (1) 로지스틱 회귀(표준화 버전, RF/XGB/LGBM과 동일 조건 비교용) ---
    log("로지스틱 회귀(표준화 버전) 학습 중...")
    pred_tr_lr, pred_val_lr = train_simple_logistic(X_tr_raw, X_val_raw, y_tr, y_val)
    results["로지스틱회귀(표준화)"] = {
        "Train": evaluate(y_tr, pred_tr_lr), "Valid": evaluate(y_val, pred_val_lr),
    }
    roc_curves["로지스틱회귀"] = (y_val, pred_val_lr)

    # --- (2) RandomForest ---
    log("RandomForest 학습 중...")
    t1 = time.time()
    rf_model, pred_tr_rf, pred_val_rf = train_random_forest(X_tr_raw, X_val_raw, y_tr)
    results["RandomForest"] = {
        "Train": evaluate(y_tr, pred_tr_rf), "Valid": evaluate(y_val, pred_val_rf),
        "학습시간(초)": round(time.time() - t1, 1),
    }
    roc_curves["RandomForest"] = (y_val, pred_val_rf)

    # --- (3) XGBoost ---
    log("XGBoost 학습 중...")
    t1 = time.time()
    xgb_model, pred_tr_xgb, pred_val_xgb = train_xgboost(X_tr_raw, X_val_raw, y_tr)
    results["XGBoost"] = {
        "Train": evaluate(y_tr, pred_tr_xgb), "Valid": evaluate(y_val, pred_val_xgb),
        "학습시간(초)": round(time.time() - t1, 1),
    }
    roc_curves["XGBoost"] = (y_val, pred_val_xgb)

    # --- (4) LightGBM ---
    log("LightGBM 학습 중...")
    t1 = time.time()
    lgbm_model, pred_tr_lgb, pred_val_lgb = train_lightgbm(X_tr_raw, X_val_raw, y_tr)
    results["LightGBM"] = {
        "Train": evaluate(y_tr, pred_tr_lgb), "Valid": evaluate(y_val, pred_val_lgb),
        "학습시간(초)": round(time.time() - t1, 1),
    }
    roc_curves["LightGBM"] = (y_val, pred_val_lgb)

    # --- 결과 비교표 ---
    rows = []
    for name, r in results.items():
        row = {"모델": name}
        for split in ("Train", "Valid"):
            for metric, v in r[split].items():
                row[f"{split}_{metric}"] = round(v, 4)
        row["학습시간(초)"] = r.get("학습시간(초)", "-")
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(f"{OUT}/모델비교_LR_RF_XGB_LGBM.csv", index=False, encoding=ENC)
    log("=== 모델 비교 결과 ===")
    print(comparison.to_string(index=False))

    plot_roc(roc_curves, f"{OUT}/모델비교_ROC_curve.png")

    # --- 변수중요도 비교 (RF/XGB/LGBM, 상위 15개) ---
    imp_df = pd.DataFrame({
        "컬럼명": feat_cols,
        "RandomForest_중요도": rf_model.feature_importances_,
        "XGBoost_중요도": xgb_model.feature_importances_,
        "LightGBM_중요도": lgbm_model.feature_importances_,
    })
    # 모델마다 스케일이 다르므로 0~1 정규화 후 비교
    for c in ["RandomForest_중요도", "XGBoost_중요도", "LightGBM_중요도"]:
        imp_df[c] = imp_df[c] / imp_df[c].sum()
    imp_df["평균순위점수"] = imp_df[["RandomForest_중요도", "XGBoost_중요도", "LightGBM_중요도"]].mean(axis=1)
    imp_df = imp_df.sort_values("평균순위점수", ascending=False).reset_index(drop=True)
    imp_df.head(15).to_csv(f"{OUT}/변수중요도_비교_top15.csv", index=False, encoding=ENC)
    log("변수중요도 상위 15개 저장 완료")

    log(f"전체 소요시간: {time.time()-t0:.1f}초")
