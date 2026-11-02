# -*- coding: utf-8 -*-
"""
STEP 7. 인코딩·스케일링 + 모델 비교 (한 번에 실행)
- 목적: 07번(파생변수)까지 완성된 피처테이블을 모델 계열별로 서로 다른 방식으로
  인코딩·스케일링하고, 곧바로 그 결과로 5개 모델(RandomForest/XGBoost/LightGBM/
  로지스틱(원핫+표준화)/WoE-로지스틱)을 학습·비교까지 한 번에 끝낸다.

모델별 요구사항:
  - 트리 계열(RandomForest/XGBoost/LightGBM/CatBoost) : 라벨인코딩만으로 충분, 스케일링 불필요
      (분기 기반 모델은 절대적 크기·분포에 영향받지 않고 순서만 보존되면 됨)
  - 선형/SVM/KNN/MLP : 원-핫인코딩 + StandardScaler 필수
      (선형모델은 라벨인코딩의 임의적 순서를 크기 관계로 잘못 해석할 위험이 있고,
       거리/내적 기반 모델은 스케일이 다르면 범위가 큰 변수가 부당하게 지배함)
  - 신용평가 특화(WoE) : Weight of Evidence 치환 하나로 인코딩+스케일링을 동시에 해결
      (모든 변수가 log-odds 스케일의 연속값이 되므로 원-핫도 스케일러도 따로 필요 없고,
       "이 구간/범주가 얼마나 위험한가"를 그대로 계수로 해석할 수 있는 신용평가 업계 표준.
       home_credit_credit_grading_pipeline.py의 v1 스코어카드와 동일한 수식·스무딩을 따름)
  - 05/05b/05c번은 각 스크립트 안에서 대충 인코딩(트리=라벨인코딩, 로지스틱=범주형 아예
    제외하고 수치형만 사용)해서 비교한 거라 로지스틱 쪽이 불리한 비교였다. 여기서는 모델
    계열별로 제대로 인코딩·스케일링한 데이터를 각 모델에 맞게 넣어서 공정하게 비교한다.

컬럼 3분류 (06번 classify_columns와 같은 판단 기준을 한 단계 더 세분화):
  - char_cat_cols    : 숫자가 아닌 dtype (직업종류/최종학력/성별 등 문자 범주형)
  - binary_flag_cols : 숫자 dtype인데 고유값 2개 이하 (서류제출여부_*, HAS_* 등 0/1 플래그)
  - numeric_cols     : 숫자 dtype이고 고유값 3개 이상 (진짜 연속형 수치)

입력: full_feature_table_derived.pkl (07번 / 마스터 통합 스크립트의 출력)
출력:
  - tree_encoded_train/valid.pkl, linear_encoded_train/valid.pkl, woe_encoded_train/valid.pkl
    (인코딩 결과 자체도 저장해둬서 나중에 다시 필요할 때 재사용 가능)
  - iv_scores.csv (WoE 단계에서 계산되는 변수별 Information Value)
  - 모델비교_LR_RF_XGB_LGBM_WoE_인코딩버전.csv, 모델비교_ROC_curve_인코딩버전.png
  - 변수중요도_비교_top15_인코딩버전.csv (트리 3종 기준)
  - WoE_로지스틱_계수_top15.csv (WoE 스코어카드 해석용 계수)

주의: 인코더(라벨 매핑 / 원-핫 카테고리 목록 / 스케일러 평균·표준편차 / WoE 구간표)는
전부 train에서만 학습(fit)하고 valid에는 그대로 적용(transform)만 한다. valid에만 있는
새 카테고리는 원-핫에서는 전부 0으로, WoE에서는 "OTHER"(또는 평균 WoE)로 처리해서
데이터 누수를 막는다. 트리/선형/WoE 세 버전 모두 같은 seed(RANDOM_STATE=42)로 나눈
동일한 train/valid 사람들이라 다섯 모델을 공정하게 비교할 수 있다.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UP = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋"
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42

SRC_PKL = f"{OUT}/full_feature_table_derived.pkl"  # 07번 출력(파생변수 포함 최종 마스터)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include="int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include="float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df


# ---------------------------------------------------------------------------
# 0) 컬럼 3분류 + train/valid 분할
# ---------------------------------------------------------------------------
def classify_columns(df: pd.DataFrame, exclude=("고유ID", TARGET, "IS_TRAIN")):
    char_cat_cols, binary_flag_cols, numeric_cols = [], [], []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            char_cat_cols.append(c)
        elif s.nunique(dropna=True) <= 2:
            binary_flag_cols.append(c)
        else:
            numeric_cols.append(c)
    return char_cat_cols, binary_flag_cols, numeric_cols


def stratified_split(train, valid_ratio=0.2, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    idx0 = train.index[train[TARGET] == 0].to_numpy().copy()
    idx1 = train.index[train[TARGET] == 1].to_numpy().copy()
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0, n1 = int(len(idx0) * valid_ratio), int(len(idx1) * valid_ratio)
    val_idx = np.concatenate([idx0[:n0], idx1[:n1]])
    tr_idx = np.concatenate([idx0[n0:], idx1[n1:]])
    return train.loc[tr_idx].reset_index(drop=True), train.loc[val_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1) 트리 계열용: 라벨인코딩 (문자 범주형만 변환, 0/1 플래그·수치형은 그대로 + 스케일 X)
# ---------------------------------------------------------------------------
def encode_for_tree(tr: pd.DataFrame, val: pd.DataFrame, char_cat_cols: list):
    tr, val = tr.copy(), val.copy()
    for c in char_cat_cols:
        cats = pd.Categorical(pd.concat([tr[c], val[c]], axis=0).fillna("MISSING")).categories
        mapping = {cat: i for i, cat in enumerate(cats)}
        tr[c] = tr[c].fillna("MISSING").map(mapping).astype("int16")
        val[c] = val[c].fillna("MISSING").map(mapping).astype("int16")
    log(f"[트리용] 라벨인코딩 완료 (문자 범주형 {len(char_cat_cols)}개, 나머지는 그대로/스케일 없음)")
    return tr, val


# ---------------------------------------------------------------------------
# 2) 선형/SVM/KNN/MLP용: 원-핫인코딩(train 카테고리 기준 고정) + StandardScaler(train 기준 학습)
# ---------------------------------------------------------------------------
def encode_for_linear(tr: pd.DataFrame, val: pd.DataFrame,
                       char_cat_cols: list, binary_flag_cols: list, numeric_cols: list):
    tr, val = tr.copy(), val.copy()

    for c in char_cat_cols:
        train_cats = tr[c].fillna("MISSING").astype(str).unique().tolist()
        tr[c] = pd.Categorical(tr[c].fillna("MISSING").astype(str), categories=train_cats)
        val[c] = pd.Categorical(val[c].fillna("MISSING").astype(str), categories=train_cats)

    tr_dummies = pd.get_dummies(tr[char_cat_cols], prefix=char_cat_cols) if char_cat_cols else pd.DataFrame(index=tr.index)
    val_dummies = pd.get_dummies(val[char_cat_cols], prefix=char_cat_cols) if char_cat_cols else pd.DataFrame(index=val.index)
    val_dummies = val_dummies.reindex(columns=tr_dummies.columns, fill_value=0)

    med = tr[numeric_cols].median()
    tr_num = tr[numeric_cols].fillna(med)
    val_num = val[numeric_cols].fillna(med)
    mu, sigma = tr_num.mean(), tr_num.std().replace(0, 1)
    tr_num_scaled = (tr_num - mu) / sigma
    val_num_scaled = (val_num - mu) / sigma

    tr_final = pd.concat([
        tr[binary_flag_cols].reset_index(drop=True),
        tr_num_scaled.reset_index(drop=True),
        tr_dummies.reset_index(drop=True),
    ], axis=1)
    val_final = pd.concat([
        val[binary_flag_cols].reset_index(drop=True),
        val_num_scaled.reset_index(drop=True),
        val_dummies.reset_index(drop=True),
    ], axis=1)

    log(f"[선형용] 원-핫 {len(char_cat_cols)}개 컬럼 -> {tr_dummies.shape[1]}개 더미, "
        f"StandardScaler {len(numeric_cols)}개 적용 -> 최종 {tr_final.shape[1]}개 피처")
    return tr_final, val_final


# ---------------------------------------------------------------------------
# 3) WoE(Weight of Evidence) 인코딩 - 신용평가 스코어카드 표준 기법
#    WoE = ln(정상비율 / 불량비율), IV = sum((정상비율-불량비율) * WoE)
# ---------------------------------------------------------------------------
def _woe_iv(good_cnt, bad_cnt, n_good, n_bad):
    good_dist = (good_cnt + 0.5) / (n_good + 0.5 * len(good_cnt))
    bad_dist = (bad_cnt + 0.5) / (n_bad + 0.5 * len(bad_cnt))
    woe = np.log(good_dist / bad_dist)
    iv = np.sum((good_dist - bad_dist) * woe)
    return woe, iv


def fit_woe_bins(tr: pd.DataFrame, char_cat_cols: list, binary_flag_cols: list,
                  numeric_cols: list, n_bins=10):
    y = tr[TARGET].astype(int).values
    n_good, n_bad = (y == 0).sum(), (y == 1).sum()
    bin_tables, iv_scores = {}, {}

    for c in numeric_cols:
        s = tr[c]
        nonnull = s.dropna()
        if nonnull.nunique() < 2:
            continue
        try:
            bins = pd.qcut(nonnull, q=n_bins, duplicates="drop")
        except Exception:
            continue
        edges = bins.cat.categories
        if len(edges) < 2:
            continue
        cat_labeled = pd.cut(s, bins=[-np.inf] + [e.right for e in edges[:-1]] + [np.inf]).astype(object)
        cat_labeled = cat_labeled.mask(s.isna(), "MISSING")
        grp = pd.DataFrame({"bin": cat_labeled, "y": y})
        counts = grp.groupby("bin", observed=True)["y"].agg(["count", "sum"])
        woe, iv = _woe_iv((counts["count"] - counts["sum"]).values, counts["sum"].values, n_good, n_bad)
        bin_tables[c] = {"type": "numeric", "edges": [float(e.right) for e in edges[:-1]], "woe": woe.tolist()}
        iv_scores[c] = float(iv)

    for c in char_cat_cols:
        s = tr[c].fillna("MISSING")
        top_cats = s.value_counts().head(15).index.tolist()
        s2 = s.where(s.isin(top_cats), "OTHER")
        grp = pd.DataFrame({"bin": s2, "y": y})
        counts = grp.groupby("bin", observed=True)["y"].agg(["count", "sum"])
        woe, iv = _woe_iv((counts["count"] - counts["sum"]).values, counts["sum"].values, n_good, n_bad)
        bin_tables[c] = {"type": "categorical", "categories": counts.index.tolist(), "woe": woe.tolist()}
        iv_scores[c] = float(iv)

    for c in binary_flag_cols:
        s = tr[c]
        grp = pd.DataFrame({"bin": s, "y": y})
        counts = grp.groupby("bin", observed=True)["y"].agg(["count", "sum"])
        if len(counts) < 2:
            continue
        woe, iv = _woe_iv((counts["count"] - counts["sum"]).values, counts["sum"].values, n_good, n_bad)
        bin_tables[c] = {"type": "categorical", "categories": counts.index.tolist(), "woe": woe.tolist()}
        iv_scores[c] = float(iv)

    return bin_tables, pd.Series(iv_scores).sort_values(ascending=False)


def apply_woe(df: pd.DataFrame, feature: str, bin_tables: dict) -> np.ndarray:
    info = bin_tables[feature]
    s = df[feature]
    if info["type"] == "numeric":
        edges, woe = info["edges"], info["woe"]
        idx = np.searchsorted(edges, s.values.astype(float), side="right")
        idx = np.where(s.isna().values, len(woe) - 1, np.clip(idx, 0, len(woe) - 1))
        return np.array(woe)[idx]
    else:
        cats, woe = info["categories"], info["woe"]
        cat_to_woe = dict(zip(cats, woe))
        default = woe[cats.index("OTHER")] if "OTHER" in cats else float(np.mean(woe))
        s_filled = s.fillna("MISSING") if not pd.api.types.is_numeric_dtype(s) else s
        if "OTHER" in cats:
            s_filled = s_filled.where(s_filled.isin(cats), "OTHER")
        return s_filled.map(cat_to_woe).fillna(default).values


def build_woe_table(df: pd.DataFrame, features: list, bin_tables: dict) -> pd.DataFrame:
    return pd.DataFrame({f: apply_woe(df, f, bin_tables) for f in features}, index=df.index)


# ---------------------------------------------------------------------------
# 공통 평가 / ROC
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
    plt.title("모델별 ROC Curve 비교 - 08번 인코딩 버전 (Valid set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    t0 = time.time()
    results, roc_curves = {}, {}

    # ---- 0) 로드 + 컬럼 분류 + 분할 ----
    log("=" * 70)
    log("데이터 로드 및 준비")
    df = pd.read_pickle(SRC_PKL)
    df = downcast(df)
    log(f"마스터 파일 로드: {df.shape}")

    char_cat_cols, binary_flag_cols, numeric_cols = classify_columns(df)
    log(f"컬럼 분류 완료 -> 문자범주형 {len(char_cat_cols)}개 / 0·1플래그 {len(binary_flag_cols)}개 "
        f"/ 수치형 {len(numeric_cols)}개")

    train = df[df["IS_TRAIN"] == 1].copy()
    del df
    gc.collect()
    train[TARGET] = train[TARGET].astype(int)
    tr, val = stratified_split(train)
    del train
    gc.collect()
    log(f"분할 완료 -> train: {tr.shape}, valid: {val.shape}, "
        f"부도율 train={tr[TARGET].mean():.4f} / valid={val[TARGET].mean():.4f}")
    y_tr, y_val = tr[TARGET].astype(int).values, val[TARGET].astype(int).values

    # ---- 1) 트리 계열: 라벨인코딩 -> RF/XGB/LGBM 학습 ----
    log("=" * 70)
    log("1/3 트리 계열 (라벨인코딩)")
    tr_tree, val_tree = encode_for_tree(tr, val, char_cat_cols)
    tr_tree.to_pickle(f"{OUT}/tree_encoded_train.pkl")
    val_tree.to_pickle(f"{OUT}/tree_encoded_valid.pkl")

    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols_tree = [c for c in tr_tree.columns if c not in exclude]
    X_tr_tree, X_val_tree = tr_tree[feat_cols_tree], val_tree[feat_cols_tree]
    med = X_tr_tree.median(numeric_only=True)
    X_tr_rf, X_val_rf = X_tr_tree.fillna(med), X_val_tree.fillna(med)  # RF만 결측 허용 안 함

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
    xgb.fit(X_tr_tree, y_tr)
    p_tr, p_val = xgb.predict_proba(X_tr_tree)[:, 1], xgb.predict_proba(X_val_tree)[:, 1]
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
    lgbm.fit(X_tr_tree, y_tr)
    p_tr, p_val = lgbm.predict_proba(X_tr_tree)[:, 1], lgbm.predict_proba(X_val_tree)[:, 1]
    results["LightGBM"] = {"Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val),
                            "학습시간(초)": round(time.time() - t1, 1)}
    roc_curves["LightGBM"] = (y_val, p_val)

    imp_df = pd.DataFrame({
        "컬럼명": feat_cols_tree,
        "RandomForest_중요도": rf.feature_importances_,
        "XGBoost_중요도": xgb.feature_importances_,
        "LightGBM_중요도": lgbm.feature_importances_,
    })
    for c in ["RandomForest_중요도", "XGBoost_중요도", "LightGBM_중요도"]:
        imp_df[c] = imp_df[c] / imp_df[c].sum()
    imp_df["평균순위점수"] = imp_df[["RandomForest_중요도", "XGBoost_중요도", "LightGBM_중요도"]].mean(axis=1)
    imp_df = imp_df.sort_values("평균순위점수", ascending=False).reset_index(drop=True)
    imp_df.head(15).to_csv(f"{OUT}/변수중요도_비교_top15_인코딩버전.csv", index=False, encoding=ENC)

    del tr_tree, val_tree, X_tr_tree, X_val_tree, X_tr_rf, X_val_rf, rf, xgb, lgbm
    gc.collect()
    log("1/3 완료 (트리 3종)")

    # ---- 2) 선형/SVM/KNN/MLP용: 원-핫 + StandardScaler -> 로지스틱 학습 ----
    log("=" * 70)
    log("2/3 로지스틱(일반) (원-핫 + StandardScaler)")
    tr_lin, val_lin = encode_for_linear(tr, val, char_cat_cols, binary_flag_cols, numeric_cols)
    tr_lin[TARGET] = y_tr
    val_lin[TARGET] = y_val
    tr_lin.to_pickle(f"{OUT}/linear_encoded_train.pkl")
    val_lin.to_pickle(f"{OUT}/linear_encoded_valid.pkl")

    feat_cols_lin = [c for c in tr_lin.columns if c != TARGET]
    X_tr_lin, X_val_lin = tr_lin[feat_cols_lin].values, val_lin[feat_cols_lin].values

    # class_weight="balanced": 부도율이 8%대로 불균형해서 이거 없이 threshold=0.5로 예측하면
    # 트리모델(class_weight/scale_pos_weight로 이미 불균형 보정됨)과 달리 로지스틱은 거의
    # 항상 "정상"으로만 예측해 Recall이 0에 가깝게 나온다. RF/XGB/LGBM과 공정하게 비교하려면
    # 로지스틱에도 같은 취지의 불균형 보정이 필요함.
    model_lin = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    model_lin.fit(X_tr_lin, y_tr)
    p_tr, p_val = model_lin.predict_proba(X_tr_lin)[:, 1], model_lin.predict_proba(X_val_lin)[:, 1]
    results["로지스틱(원핫+표준화)"] = {"Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val)}
    roc_curves["로지스틱(원핫+표준화)"] = (y_val, p_val)

    del tr_lin, val_lin, X_tr_lin, X_val_lin, model_lin
    gc.collect()
    log("2/3 완료 (로지스틱 일반)")

    # ---- 3) WoE(신용평가 특화)용: WoE 치환 -> 로지스틱 학습 ----
    log("=" * 70)
    log("3/3 WoE-로지스틱(신용평가 스코어카드)")
    all_features = numeric_cols + char_cat_cols + binary_flag_cols
    bin_tables, iv_scores = fit_woe_bins(tr, char_cat_cols, binary_flag_cols, numeric_cols)
    fitted_features = list(bin_tables.keys())
    log(f"WoE 구간표 학습 완료: {len(fitted_features)}/{len(all_features)}개 변수")

    tr_woe = build_woe_table(tr, fitted_features, bin_tables)
    val_woe = build_woe_table(val, fitted_features, bin_tables)
    tr_woe[TARGET] = y_tr
    val_woe[TARGET] = y_val
    tr_woe.to_pickle(f"{OUT}/woe_encoded_train.pkl")
    val_woe.to_pickle(f"{OUT}/woe_encoded_valid.pkl")
    iv_scores.to_csv(f"{OUT}/iv_scores.csv", header=["IV"], encoding=ENC)

    feat_cols_woe = [c for c in tr_woe.columns if c != TARGET]
    X_tr_woe, X_val_woe = tr_woe[feat_cols_woe].values, val_woe[feat_cols_woe].values

    model_woe = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    model_woe.fit(X_tr_woe, y_tr)
    p_tr, p_val = model_woe.predict_proba(X_tr_woe)[:, 1], model_woe.predict_proba(X_val_woe)[:, 1]
    results["WoE-로지스틱(스코어카드)"] = {"Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val)}
    roc_curves["WoE-로지스틱(스코어카드)"] = (y_val, p_val)

    coef_df = pd.DataFrame({"컬럼명": feat_cols_woe, "계수": model_woe.coef_[0]})
    coef_df["절대값"] = coef_df["계수"].abs()
    coef_df = coef_df.sort_values("절대값", ascending=False).drop(columns="절대값").reset_index(drop=True)
    coef_df.head(15).to_csv(f"{OUT}/WoE_로지스틱_계수_top15.csv", index=False, encoding=ENC)

    del tr, val, tr_woe, val_woe, X_tr_woe, X_val_woe, model_woe
    gc.collect()
    log("3/3 완료 (WoE 로지스틱)")

    # ---- 결과 정리 ----
    log("=" * 70)
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
    log("=== 모델 비교 결과 (08번 인코딩 버전, 5개 모델) ===")
    print(comparison.to_string(index=False))

    plot_roc(roc_curves, f"{OUT}/모델비교_ROC_curve_인코딩버전.png")
    log(f"IV 상위 10개 변수:\n{iv_scores.head(10).to_string()}")
    log(f"전체 소요시간: {time.time()-t0:.1f}초")
