# -*- coding: utf-8 -*-
"""
STEP 12. 교차검증 및 하이퍼파라미터 튜닝
- 신규 추가 이유: 지금까지(05b/08/11/12번)는 데이터를 딱 한 번만 train/valid로 나눠서
  모델을 비교했다. 이 한 번의 분할에서 우연히 특정 모델이 유리/불리하게 나뉠 수 있어
  공정한 비교라고 보기 어렵다. StratifiedKFold(cv=5)로 5번 다르게 나눠서 평균±표준편차를
  보면 훨씬 신뢰할 수 있는 비교가 된다.

PART 1) StratifiedKFold(cv=5) 교차검증 비교
  - 최종 후보로 좁힌 4개 모델(XGBoost, LightGBM, CatBoost, LogisticRegression)을 대상으로,
    08/07번이 만든 계열별 인코딩을 그대로 재사용해 5-fold CV의 평균 AUC ± 표준편차를 구한다.
  - 모델 파라미터는 12번에서 쓴 값 그대로 고정(아직 튜닝 전, "기본형" 상태의 공정 비교).

PART 2) 상위 후보 하이퍼파라미터 튜닝
  - PART 1의 평균 CV AUC 기준 상위 TOP_N_TUNE개 모델만 골라 RandomizedSearchCV로 튜닝한다.
    (GridSearchCV는 이 정도 데이터 규모 x 파라미터 조합 수에서는 계산량이 비현실적으로 커서,
    RandomizedSearchCV로 무작위 일부 조합만 표본 추출해 탐색한다. 더 정교하게 하고 싶으면
    Optuna 같은 베이지안 최적화 라이브러리로 대체할 수 있지만 추가 설치가 필요해서 여기서는
    sklearn 기본 기능만으로 구현했다.)
  - 튜닝 후 best_estimator_를 08/12번과 동일한 valid set에 다시 적용해서, 튜닝 전(기존 결과)과
    튜닝 후 성능을 직접 비교할 수 있게 한다.

주의(중요): CatBoost는 1회 학습에도 5분 이상 걸리는 모델이라, 5-fold CV(PART 1)만 해도
  CatBoost 한 모델에서 약 25~30분 정도 걸릴 수 있다. PART 2 튜닝은 조합 수 x cv=3번씩 다시
  학습하므로 CatBoost가 상위 후보에 포함되면 1시간 이상 걸릴 수 있다. 아래
  RANDOM_SEARCH_N_ITER 값을 낮추면 더 빨리 끝나니, 너무 오래 걸리면 줄여서 다시 실행해도 된다.

입력: tree_encoded_train/valid.pkl, linear_encoded_train/valid.pkl (08번 출력),
      full_feature_table_derived.pkl (07번 출력, CatBoost 전용)
출력: CV_비교결과_5fold.csv, 하이퍼파라미터튜닝_결과.csv
"""
import time
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError(
        "catboost 패키지가 필요합니다. 터미널에서 'conda activate homecredit' 후 "
        "'conda install -c conda-forge catboost -y' 실행하고 Spyder 커널을 재시작한 뒤 다시 시도하세요."
    ) from e


class _CatBoostClassifierSK(ClassifierMixin, CatBoostClassifier, BaseEstimator):
    """catboost가 아직 scikit-learn 1.6+의 새 __sklearn_tags__ 프로토콜을 지원하지 않아서
    (catboost GitHub Issue #2955, 아직 미해결) cross_val_score/RandomizedSearchCV가 내부에서
    모델을 clone()할 때 AttributeError가 난다. ClassifierMixin과 BaseEstimator를 뒤에
    상속시켜 __sklearn_tags__를 자동으로 채워주는 방식으로 우회한다(에러 메시지가 직접
    권장하는 해결 방법). __init__/fit/predict 등은 전부 CatBoostClassifier 그대로 사용됨."""
    pass

UP = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋"
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42
CV_FOLDS = 5              # PART 1: 모델 비교용
TUNE_CV_FOLDS = 3         # PART 2: 튜닝용(속도 위해 5 대신 3)
TOP_N_TUNE = 3            # PART 1 결과 상위 몇 개를 튜닝할지
RANDOM_SEARCH_N_ITER = {  # 모델별 랜덤서치 시도 횟수 (CatBoost는 느려서 적게 잡음)
    "XGBoost": 25,
    "LightGBM": 25,
    "CatBoost": 8,
    "LogisticRegression": 20,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def split_same_as_before(train, valid_ratio=0.2, seed=RANDOM_STATE):
    """05b/08/11/12번과 동일한 방식(같은 seed)으로 train/valid를 나눠 동일 인물로 맞춘다."""
    rng = np.random.default_rng(seed)
    idx0 = train.index[train[TARGET] == 0].to_numpy().copy()
    idx1 = train.index[train[TARGET] == 1].to_numpy().copy()
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0, n1 = int(len(idx0) * valid_ratio), int(len(idx1) * valid_ratio)
    val_idx = np.concatenate([idx0[:n0], idx1[:n1]])
    tr_idx = np.concatenate([idx0[n0:], idx1[n1:]])
    return train.loc[tr_idx].reset_index(drop=True), train.loc[val_idx].reset_index(drop=True)


if __name__ == "__main__":
    t0 = time.time()

    # =====================================================================
    # 데이터 로드 (08/07번 출력 재사용)
    # =====================================================================
    log("=" * 70)
    log("데이터 로드")
    tr_tree = pd.read_pickle(f"{OUT}/tree_encoded_train.pkl")
    val_tree = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl")
    tr_lin = pd.read_pickle(f"{OUT}/linear_encoded_train.pkl")
    val_lin = pd.read_pickle(f"{OUT}/linear_encoded_valid.pkl")
    df_cb = pd.read_pickle(f"{OUT}/full_feature_table_derived.pkl")

    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols_tree = [c for c in tr_tree.columns if c not in exclude]
    feat_cols_lin = [c for c in tr_lin.columns if c != TARGET]

    med = tr_tree[feat_cols_tree].median(numeric_only=True)
    X_tree = tr_tree[feat_cols_tree].fillna(med).values
    y = tr_tree[TARGET].astype(int).values
    X_val_tree = val_tree[feat_cols_tree].fillna(med).values
    y_val = val_tree[TARGET].astype(int).values

    X_lin = tr_lin[feat_cols_lin].values
    X_val_lin = val_lin[feat_cols_lin].values

    del tr_tree, val_tree, tr_lin, val_lin
    gc.collect()

    n_pos, n_neg = y.sum(), len(y) - y.sum()
    spw = n_neg / n_pos

    # CatBoost용 원본 범주형 데이터 (07번 파생변수 테이블에서 동일 split 재현)
    train_cb_all = df_cb[df_cb["IS_TRAIN"] == 1].copy()
    del df_cb
    gc.collect()
    train_cb_all[TARGET] = train_cb_all[TARGET].astype(int)
    char_cat_cols = [c for c in train_cb_all.columns if c not in exclude
                      and (not pd.api.types.is_numeric_dtype(train_cb_all[c])
                           or pd.api.types.is_bool_dtype(train_cb_all[c]))]
    feat_cols_cb = [c for c in train_cb_all.columns if c not in exclude]
    for c in char_cat_cols:
        train_cb_all[c] = train_cb_all[c].fillna("MISSING").astype(str)
    cat_idx = [feat_cols_cb.index(c) for c in char_cat_cols]

    tr_cb, val_cb = split_same_as_before(train_cb_all)
    del train_cb_all
    gc.collect()
    X_cb = tr_cb[feat_cols_cb]
    y_cb = tr_cb[TARGET].values
    X_val_cb = val_cb[feat_cols_cb]
    y_val_cb = val_cb[TARGET].values
    del tr_cb, val_cb
    gc.collect()

    log(f"데이터 준비 완료: train {len(y)}행 (부도율 {y.mean():.4f})")

    # =====================================================================
    # 모델 정의 (12번과 동일한 기본 파라미터로 고정 - 튜닝 전 비교용)
    # =====================================================================
    def make_xgb():
        return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="auc",
                              random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")

    def make_lgbm():
        return LGBMClassifier(n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                               random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)

    def make_cat():
        return _CatBoostClassifierSK(iterations=500, depth=6, learning_rate=0.05, cat_features=cat_idx,
                                      auto_class_weights="Balanced", random_state=RANDOM_STATE,
                                      verbose=False, thread_count=1)

    def make_lr():
        return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE)

    # =====================================================================
    # PART 1. StratifiedKFold(cv=5) 모델 비교
    # =====================================================================
    log("=" * 70)
    log(f"PART 1. StratifiedKFold(cv={CV_FOLDS}) 모델 비교 시작 "
        f"(CatBoost 포함이라 전체 20~30분 이상 걸릴 수 있음)")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_rows = []

    def run_cv(name, model_fn, X, y_):
        t1 = time.time()
        scores = cross_val_score(model_fn(), X, y_, scoring="roc_auc", cv=skf, n_jobs=1)
        elapsed = time.time() - t1
        cv_rows.append({
            "모델": name, "CV_AUC_평균": round(scores.mean(), 4),
            "CV_AUC_표준편차": round(scores.std(), 4),
            "Fold별_AUC": [round(s, 4) for s in scores],
            "소요시간(초)": round(elapsed, 1),
        })
        log(f"[{name}] CV AUC = {scores.mean():.4f} ± {scores.std():.4f} ({elapsed:.1f}초)")

    log("-" * 70); log("XGBoost 5-fold CV...")
    run_cv("XGBoost", make_xgb, X_tree, y)

    log("-" * 70); log("LightGBM 5-fold CV...")
    run_cv("LightGBM", make_lgbm, X_tree, y)

    log("-" * 70); log("LogisticRegression 5-fold CV...")
    run_cv("LogisticRegression", make_lr, X_lin, y)

    log("-" * 70); log("CatBoost 5-fold CV... (가장 오래 걸림)")
    run_cv("CatBoost", make_cat, X_cb, y_cb)

    cv_result = pd.DataFrame(cv_rows).sort_values("CV_AUC_평균", ascending=False).reset_index(drop=True)
    cv_result_save = cv_result.copy()
    cv_result_save["Fold별_AUC"] = cv_result_save["Fold별_AUC"].apply(lambda x: str(x))
    cv_result_save.to_csv(f"{OUT}/CV_비교결과_5fold.csv", index=False, encoding=ENC)
    log("=== PART 1 결과 (CV AUC 평균 내림차순) ===")
    print(cv_result_save.to_string(index=False))

    # =====================================================================
    # PART 2. 상위 후보 하이퍼파라미터 튜닝 (RandomizedSearchCV)
    # =====================================================================
    top_models = cv_result["모델"].head(TOP_N_TUNE).tolist()
    log("=" * 70)
    log(f"PART 2. 상위 {TOP_N_TUNE}개 모델 튜닝 대상: {top_models}")

    param_dists = {
        "XGBoost": {
            "n_estimators": randint(200, 800),
            "max_depth": randint(3, 9),
            "learning_rate": uniform(0.01, 0.19),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 10),
        },
        "LightGBM": {
            "n_estimators": randint(200, 800),
            "num_leaves": randint(15, 127),
            "max_depth": [-1, 4, 6, 8, 10, 12],
            "learning_rate": uniform(0.01, 0.19),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        },
        "CatBoost": {
            "iterations": [300, 500, 800],
            "depth": randint(4, 10),
            "learning_rate": uniform(0.01, 0.19),
            "l2_leaf_reg": uniform(1, 9),
        },
        "LogisticRegression": {
            "C": loguniform(1e-3, 1e2),
            "penalty": ["l1", "l2"],
        },
    }

    model_data = {
        "XGBoost": (make_xgb, X_tree, y, X_val_tree, y_val),
        "LightGBM": (make_lgbm, X_tree, y, X_val_tree, y_val),
        "CatBoost": (make_cat, X_cb, y_cb, X_val_cb, y_val_cb),
        "LogisticRegression": (make_lr, X_lin, y, X_val_lin, y_val),
    }

    tune_cv = StratifiedKFold(n_splits=TUNE_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tune_rows = []
    for name in top_models:
        model_fn, X_, y_, X_v, y_v = model_data[name]
        base_model = model_fn()
        if name == "LogisticRegression":
            base_model.set_params(solver="liblinear")  # l1/l2 둘 다 지원하는 솔버로 고정
        log("-" * 70)
        log(f"{name} RandomizedSearchCV 튜닝 시작 (n_iter={RANDOM_SEARCH_N_ITER[name]}, "
            f"cv={TUNE_CV_FOLDS})...")
        t1 = time.time()
        search = RandomizedSearchCV(
            base_model, param_distributions=param_dists[name],
            n_iter=RANDOM_SEARCH_N_ITER[name], scoring="roc_auc",
            cv=tune_cv, random_state=RANDOM_STATE, n_jobs=1, refit=True,
        )
        search.fit(X_, y_)
        elapsed = time.time() - t1

        p_val = search.best_estimator_.predict_proba(X_v)[:, 1]
        valid_auc = roc_auc_score(y_v, p_val)

        tune_rows.append({
            "모델": name,
            "튜닝전_CV_AUC": cv_result.loc[cv_result["모델"] == name, "CV_AUC_평균"].values[0],
            "튜닝후_CV_AUC": round(search.best_score_, 4),
            "튜닝후_Valid_AUC": round(valid_auc, 4),
            "최적_파라미터": str(search.best_params_),
            "소요시간(초)": round(elapsed, 1),
        })
        log(f"[{name}] 튜닝후 CV AUC={search.best_score_:.4f}, Valid AUC={valid_auc:.4f} "
            f"({elapsed:.1f}초)")
        log(f"[{name}] 최적 파라미터: {search.best_params_}")

    tune_result = pd.DataFrame(tune_rows).sort_values("튜닝후_Valid_AUC", ascending=False).reset_index(drop=True)
    tune_result.to_csv(f"{OUT}/하이퍼파라미터튜닝_결과.csv", index=False, encoding=ENC)
    log("=== PART 2 결과 ===")
    print(tune_result.to_string(index=False))

    log("=" * 70)
    log(f"전체 소요시간: {(time.time()-t0)/60:.1f}분")
