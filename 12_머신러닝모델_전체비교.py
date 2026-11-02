# -*- coding: utf-8 -*-
"""
STEP 11. 머신러닝 모델 학습 (12종 알고리즘 폭넓게 비교)
- 목적: 계열이 다른 다양한 알고리즘으로 성능을 폭넓게 비교해서, 이 데이터에 어떤 "종류"의
  모델이 잘 맞는지 파악한다 (트리 앙상블 / 선형모델 / 거리기반모델 / 범주형 특화모델).

모델 목록과 사용 데이터 (08번에서 만든 모델 계열별 인코딩을 그대로 재사용):
  [트리 계열 - tree_encoded (라벨인코딩, 스케일 없음)]
    DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, XGBClassifier, LGBMClassifier
  [선형/거리 계열 - linear_encoded (원-핫 + StandardScaler)]
    LogisticRegression, Ridge, Lasso, SVC(rbf), KNeighborsClassifier
  [범주형 특화 - full_feature_table_derived.pkl 원본 그대로]
    CatBoostClassifier (문자 범주형을 라벨인코딩하지 않고 원본 그대로 cat_features로 넘겨줘야
    CatBoost 고유의 "순서형 타겟 통계(ordered target statistics)" 방식이 제대로 작동한다.
    범주형 변수가 많은 이 데이터에서 특히 강점을 보일 것으로 기대되는 이유)

Ridge/Lasso에 대한 설명: scikit-learn의 Ridge/Lasso는 원래 회귀(연속값 예측)용 클래스라
분류에는 못 쓴다. 분류에서 같은 취지(L2/L1 정규화)를 내는 표준적인 방법을 대신 사용했다:
  - "Ridge"  -> RidgeClassifier (L2 정규화, 0/1을 -1/+1로 바꿔 회귀처럼 풀고 부호로 분류)
  - "Lasso"  -> LogisticRegression(penalty="l1") (L1 정규화 로지스틱 회귀,
                일부 계수를 정확히 0으로 만들어 변수를 자동 선택하는 Lasso의 특징을 그대로 가짐)

불균형 처리: 11번(불균형 처리 비교)에서 class_weight="balanced"가 리샘플링과 동급 성능에
가장 간단하다는 게 확인됐으므로, 여기서는 각 모델이 지원하는 방식으로 전부 불균형 보정을
켜서 비교한다 (트리=class_weight/scale_pos_weight, XGB/LGBM=scale_pos_weight,
CatBoost=auto_class_weights). 단, sklearn의 GradientBoostingClassifier와
KNeighborsClassifier는 애초에 class_weight 파라미터를 지원하지 않아 보정 없이 돌린다
(GBM은 sample_weight로 대신 보정, KNN은 weights="distance"로 부분적으로만 완화).

SVC(rbf)/KNeighborsClassifier 관련 중요 주의사항: 이 두 알고리즘은 학습 데이터 규모가
클수록 계산량이 급격히 늘어나서(SVC는 이론상 O(n^2)~O(n^3), KNN은 예측할 때마다 전체
학습 데이터와 거리를 계산) 30만 행 전체로 돌리면 며칠이 걸릴 수도 있다. 그래서 이 두
모델만 train을 무작위 서브샘플(기본 15,000행)로 줄여서 학습한다. 나머지 10개 모델과
직접적인 절대 성능 비교보다는 "이 정도 규모로 압축해도 어느 정도 판별력이 나오는가"
참고용으로 보는 게 맞다.

입력: tree_encoded_train/valid.pkl, linear_encoded_train/valid.pkl (08번 출력),
      full_feature_table_derived.pkl (07번 출력, CatBoost 전용)
출력: 모델비교_머신러닝12종.csv, 모델비교_머신러닝12종_ROC_트리계열.png,
      모델비교_머신러닝12종_ROC_선형거리계열.png

필요 패키지: catboost (없으면 설치 필요)
  conda install -c conda-forge catboost
  또는 pip install catboost
"""
import time
import gc
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError(
        "catboost 패키지가 필요합니다. 터미널에서 'conda activate homecredit' 후 "
        "'conda install -c conda-forge catboost -y' (또는 'pip install catboost') "
        "실행하고 Spyder 커널을 재시작한 뒤 다시 시도하세요."
    ) from e

UP = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋"
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42
SVC_KNN_SUBSAMPLE_N = 15000  # SVC/KNN 전용 train 서브샘플 크기


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


def record(results, roc_curves, name, y_tr, p_tr, y_val, p_val, elapsed, threshold=0.5):
    results[name] = {
        "Train": evaluate(y_tr, p_tr, threshold), "Valid": evaluate(y_val, p_val, threshold),
        "학습시간(초)": round(elapsed, 1),
    }
    roc_curves[name] = (y_val, p_val)
    r = results[name]["Valid"]
    log(f"[{name}] Valid AUC={r['AUC']:.4f}, Recall={r['Recall']:.4f} ({elapsed:.1f}초)")


def stratified_subsample(X, y, n, seed=RANDOM_STATE):
    """SVC/KNN처럼 계산량이 큰 모델을 위해, 원래 부도율을 유지한 채 train만 축소."""
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    ratio = len(idx1) / (len(idx0) + len(idx1))
    n1 = max(1, int(n * ratio))
    n0 = n - n1
    idx0_s = rng.choice(idx0, size=min(n0, len(idx0)), replace=False)
    idx1_s = rng.choice(idx1, size=min(n1, len(idx1)), replace=False)
    idx = np.concatenate([idx0_s, idx1_s])
    rng.shuffle(idx)
    return X[idx], y[idx]


def plot_roc(curves: dict, path, title):
    plt.figure(figsize=(8, 8))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    t0 = time.time()
    results, roc_curves_tree, roc_curves_linear = {}, {}, {}

    # =========================================================================
    # A) 트리 계열 - tree_encoded (라벨인코딩, 스케일 없음)
    # =========================================================================
    log("=" * 70)
    log("A) 트리 계열 로드 (tree_encoded_*.pkl)")
    tr = pd.read_pickle(f"{OUT}/tree_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl")
    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols_tree = [c for c in tr.columns if c not in exclude]
    med = tr[feat_cols_tree].median(numeric_only=True)
    X_tr_tree = tr[feat_cols_tree].fillna(med)
    X_val_tree = val[feat_cols_tree].fillna(med)
    y_tr = tr[TARGET].astype(int).values
    y_val = val[TARGET].astype(int).values
    del tr, val
    gc.collect()
    log(f"트리용 데이터 준비 완료: train {X_tr_tree.shape}, valid {X_val_tree.shape}")

    log("-" * 70); log("DecisionTreeClassifier 학습 중...")
    t1 = time.time()
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_STATE)
    dt.fit(X_tr_tree, y_tr)
    record(results, roc_curves_tree, "DecisionTree", y_tr, dt.predict_proba(X_tr_tree)[:, 1],
           y_val, dt.predict_proba(X_val_tree)[:, 1], time.time() - t1)

    log("-" * 70); log("RandomForestClassifier 학습 중...")
    t1 = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                 class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE)
    rf.fit(X_tr_tree, y_tr)
    record(results, roc_curves_tree, "RandomForest", y_tr, rf.predict_proba(X_tr_tree)[:, 1],
           y_val, rf.predict_proba(X_val_tree)[:, 1], time.time() - t1)

    log("-" * 70); log("ExtraTreesClassifier 학습 중...")
    t1 = time.time()
    et = ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                               class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE)
    et.fit(X_tr_tree, y_tr)
    record(results, roc_curves_tree, "ExtraTrees", y_tr, et.predict_proba(X_tr_tree)[:, 1],
           y_val, et.predict_proba(X_val_tree)[:, 1], time.time() - t1)

    log("-" * 70); log("GradientBoostingClassifier 학습 중... (sklearn 구현이라 다소 느릴 수 있음)")
    t1 = time.time()
    # GradientBoostingClassifier는 class_weight 파라미터가 없어서 sample_weight로 직접 보정
    n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    sw = np.where(y_tr == 1, len(y_tr) / (2 * n_pos), len(y_tr) / (2 * n_neg))
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=RANDOM_STATE)
    gbm.fit(X_tr_tree, y_tr, sample_weight=sw)
    record(results, roc_curves_tree, "GradientBoosting", y_tr, gbm.predict_proba(X_tr_tree)[:, 1],
           y_val, gbm.predict_proba(X_val_tree)[:, 1], time.time() - t1)

    log("-" * 70); log("XGBClassifier 학습 중...")
    t1 = time.time()
    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=n_neg / n_pos,
                         eval_metric="auc", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")
    xgb.fit(X_tr_tree, y_tr)
    record(results, roc_curves_tree, "XGBoost", y_tr, xgb.predict_proba(X_tr_tree)[:, 1],
           y_val, xgb.predict_proba(X_val_tree)[:, 1], time.time() - t1)

    log("-" * 70); log("LGBMClassifier 학습 중...")
    t1 = time.time()
    lgbm = LGBMClassifier(n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, scale_pos_weight=n_neg / n_pos,
                           random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)
    lgbm.fit(X_tr_tree, y_tr)
    record(results, roc_curves_tree, "LightGBM", y_tr, lgbm.predict_proba(X_tr_tree)[:, 1],
           y_val, lgbm.predict_proba(X_val_tree)[:, 1], time.time() - t1)

    del X_tr_tree, X_val_tree, dt, rf, et, gbm, xgb, lgbm
    gc.collect()

    # =========================================================================
    # B) 선형/거리 계열 - linear_encoded (원-핫 + StandardScaler)
    # =========================================================================
    log("=" * 70)
    log("B) 선형/거리 계열 로드 (linear_encoded_*.pkl)")
    tr = pd.read_pickle(f"{OUT}/linear_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/linear_encoded_valid.pkl")
    feat_cols_lin = [c for c in tr.columns if c != TARGET]
    X_tr_lin = tr[feat_cols_lin].values
    X_val_lin = val[feat_cols_lin].values
    del tr, val
    gc.collect()
    log(f"선형용 데이터 준비 완료: train {X_tr_lin.shape}, valid {X_val_lin.shape}")

    log("-" * 70); log("LogisticRegression 학습 중...")
    t1 = time.time()
    logit = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    logit.fit(X_tr_lin, y_tr)
    record(results, roc_curves_linear, "LogisticRegression", y_tr, logit.predict_proba(X_tr_lin)[:, 1],
           y_val, logit.predict_proba(X_val_lin)[:, 1], time.time() - t1)

    log("-" * 70); log("RidgeClassifier(Ridge) 학습 중...")
    t1 = time.time()
    ridge = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    ridge.fit(X_tr_lin, y_tr)
    # RidgeClassifier는 predict_proba가 없어 decision_function(스코어, 0 기준으로 분류)을 사용.
    # AUC는 순위만 보므로 문제없고, Accuracy/Precision/Recall 계산 시 threshold를 0으로 맞춤.
    record(results, roc_curves_linear, "Ridge", y_tr, ridge.decision_function(X_tr_lin),
           y_val, ridge.decision_function(X_val_lin), time.time() - t1, threshold=0.0)

    log("-" * 70); log("LogisticRegression(L1=Lasso) 학습 중...")
    t1 = time.time()
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                class_weight="balanced", random_state=RANDOM_STATE)
    lasso.fit(X_tr_lin, y_tr)
    n_nonzero = int(np.sum(lasso.coef_[0] != 0))
    log(f"Lasso(L1) 계수 중 0이 아닌 것: {n_nonzero}/{len(feat_cols_lin)}개 (나머지는 자동으로 제외됨)")
    record(results, roc_curves_linear, "Lasso(L1)", y_tr, lasso.predict_proba(X_tr_lin)[:, 1],
           y_val, lasso.predict_proba(X_val_lin)[:, 1], time.time() - t1)

    # ---- SVC/KNN 전용 서브샘플 ----
    X_tr_sub, y_tr_sub = stratified_subsample(X_tr_lin, y_tr, SVC_KNN_SUBSAMPLE_N)
    log(f"SVC/KNN용 서브샘플: {X_tr_sub.shape} (부도율={y_tr_sub.mean():.4f}, "
        f"전체 {len(y_tr)}행 중 {SVC_KNN_SUBSAMPLE_N}행만 사용 - 계산량 문제로 축소)")

    log("-" * 70); log("SVC(rbf) 학습 중... (서브샘플 사용, 그래도 몇 분 걸릴 수 있음)")
    t1 = time.time()
    svc = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    svc.fit(X_tr_sub, y_tr_sub)
    record(results, roc_curves_linear, "SVC(rbf, 서브샘플)", y_tr_sub, svc.predict_proba(X_tr_sub)[:, 1],
           y_val, svc.predict_proba(X_val_lin)[:, 1], time.time() - t1)

    log("-" * 70); log("KNeighborsClassifier 학습 중... (서브샘플 사용)")
    t1 = time.time()
    # KNN은 class_weight를 지원하지 않음 -> weights="distance"로 가까운 이웃에 더 큰 투표권을
    # 줘서 불균형을 부분적으로만 완화 (완전한 대안은 아님, 한계로 명시)
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1)
    knn.fit(X_tr_sub, y_tr_sub)
    record(results, roc_curves_linear, "KNN(서브샘플)", y_tr_sub, knn.predict_proba(X_tr_sub)[:, 1],
           y_val, knn.predict_proba(X_val_lin)[:, 1], time.time() - t1)

    del X_tr_lin, X_val_lin, X_tr_sub, y_tr_sub, logit, ridge, lasso, svc, knn
    gc.collect()

    # =========================================================================
    # C) CatBoostClassifier - 원본 문자 범주형 그대로 (cat_features 지정)
    # =========================================================================
    log("=" * 70)
    log("C) CatBoost용 원본 데이터 로드 (full_feature_table_derived.pkl)")
    df = pd.read_pickle(f"{OUT}/full_feature_table_derived.pkl")
    char_cat_cols = [c for c in df.columns if c not in exclude
                      and (not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    feat_cols_cb = [c for c in df.columns if c not in exclude]

    train = df[df["IS_TRAIN"] == 1].copy()
    del df
    gc.collect()
    train[TARGET] = train[TARGET].astype(int)

    def _split(train, valid_ratio=0.2, seed=RANDOM_STATE):
        rng = np.random.default_rng(seed)
        idx0 = train.index[train[TARGET] == 0].to_numpy().copy()
        idx1 = train.index[train[TARGET] == 1].to_numpy().copy()
        rng.shuffle(idx0); rng.shuffle(idx1)
        n0, n1 = int(len(idx0) * valid_ratio), int(len(idx1) * valid_ratio)
        val_idx = np.concatenate([idx0[:n0], idx1[:n1]])
        tr_idx = np.concatenate([idx0[n0:], idx1[n1:]])
        return train.loc[tr_idx].reset_index(drop=True), train.loc[val_idx].reset_index(drop=True)

    tr_cb, val_cb = _split(train)  # 05b/08/11과 동일 seed -> 같은 사람들로 나뉨
    del train
    gc.collect()

    for c in char_cat_cols:  # CatBoost는 범주형에 결측이 있으면 문자열로 명시해줘야 안전함
        tr_cb[c] = tr_cb[c].fillna("MISSING").astype(str)
        val_cb[c] = val_cb[c].fillna("MISSING").astype(str)

    X_tr_cb, X_val_cb = tr_cb[feat_cols_cb], val_cb[feat_cols_cb]
    y_tr_cb, y_val_cb = tr_cb[TARGET].values, val_cb[TARGET].values
    cat_idx = [feat_cols_cb.index(c) for c in char_cat_cols]
    log(f"CatBoost용 데이터 준비 완료: train {X_tr_cb.shape}, valid {X_val_cb.shape}, "
        f"범주형 컬럼 {len(cat_idx)}개 (원본 문자열 그대로 사용)")
    del tr_cb, val_cb
    gc.collect()

    log("-" * 70); log("CatBoostClassifier 학습 중...")
    t1 = time.time()
    cat = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        cat_features=cat_idx, auto_class_weights="Balanced",
        random_state=RANDOM_STATE, verbose=False, thread_count=1,
    )
    cat.fit(X_tr_cb, y_tr_cb)
    record(results, roc_curves_tree, "CatBoost", y_tr_cb, cat.predict_proba(X_tr_cb)[:, 1],
           y_val_cb, cat.predict_proba(X_val_cb)[:, 1], time.time() - t1)

    del X_tr_cb, X_val_cb, cat
    gc.collect()

    # =========================================================================
    # 결과 정리
    # =========================================================================
    log("=" * 70)
    rows = []
    for name, r in results.items():
        row = {"모델": name}
        for split in ("Train", "Valid"):
            for metric, v in r[split].items():
                row[f"{split}_{metric}"] = round(v, 4)
        row["학습시간(초)"] = r["학습시간(초)"]
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("Valid_AUC", ascending=False).reset_index(drop=True)
    comparison.to_csv(f"{OUT}/모델비교_머신러닝12종.csv", index=False, encoding=ENC)
    log("=== 머신러닝 12종 비교 결과 (Valid AUC 내림차순) ===")
    print(comparison.to_string(index=False))

    plot_roc(roc_curves_tree, f"{OUT}/모델비교_머신러닝12종_ROC_트리계열.png",
              "ROC Curve - 트리 계열 + CatBoost (Valid set)")
    plot_roc(roc_curves_linear, f"{OUT}/모델비교_머신러닝12종_ROC_선형거리계열.png",
              "ROC Curve - 선형/거리 계열 (Valid set)")

    log(f"전체 소요시간: {time.time()-t0:.1f}초")
