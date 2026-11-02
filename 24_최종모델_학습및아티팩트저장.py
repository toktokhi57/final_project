# -*- coding: utf-8 -*-
"""
STEP 24. 최종 모델 학습 + 스코어링 아티팩트 저장 (파이프라인 통합 1/2)

배경:
- 12~23번까지 전부 "분석/검증용" 스크립트라서, 실행할 때마다 데이터를 다시 불러오고
  모델을 다시 학습했다. 이제 지금까지의 모든 결정(12번 모델 파라미터, 17번 Platt 보정,
  18번 등급구간, 23번 의사결정규칙)이 21번 검증을 통해 새 데이터에서도 일반화됨을
  확인했으니, 이 스크립트로 "진짜 최종" 모델을 한 번만 학습해서 저장해둔다.
  - 지금까지는 train만 학습에 쓰고 valid는 검증용으로 아껴뒀지만, 이제 검증이 끝났으므로
    실전 배포용 모델은 train+valid를 전부 합쳐서 학습한다(데이터를 더 많이 쓸수록 보통
    성능이 좋아진다 - 검증 목적이 사라졌으니 아껴둘 이유가 없다).
  - 이후로는 25번(homecredit_scorer.py)이 여기서 저장한 파일들만 불러와서 새 고객
    1명을 몇 초 안에 채점할 수 있다 (매번 XGBoost를 재학습할 필요 없음).

절차:
  1) full_feature_table_derived.pkl(07번 출력, 파생변수까지 포함된 원본 텍스트 상태)에서
     라벨(목표변수)이 있는 행(IS_TRAIN==1)만 사용
  2) 08번과 동일한 방식으로 컬럼을 문자범주형/이진플래그/수치형으로 분류하고, 문자범주형은
     라벨인코딩 - 이때 카테고리->정수 매핑표를 "아티팩트"로 저장해서 나중에 새 고객의
     텍스트 값(예: "여", "고졸")을 정확히 같은 정수로 바꿀 수 있게 한다
  3) 전체 라벨 데이터를 85%(학습)/15%(보정) 로 나눠 XGBoost 학습 + Platt 보정기 학습
     (17번과 동일 방식)
  4) 보정용 15%에서 등급 구간(18번 방법3=기저율중심)과 등급별 의사결정규칙(23번의
     LGD/순이익률 기반 EV, 가산금리)을 새로 계산 - 이번엔 전체 데이터 기준으로 다시
     계산하므로 이전 스크립트들의 하드코딩된 숫자에 의존하지 않는다
  5) 모델/보정기/인코딩맵/등급구간/의사결정표를 전부 하나의 pickle 번들로 저장

입력: full_feature_table_derived.pkl (07번 출력)
출력: 최종_스코어링_아티팩트.pkl (25번 homecredit_scorer.py가 그대로 불러다 씀)
"""
import time
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42
GRADES = ["A", "B", "C", "D", "E"]
LGD = 0.45
NET_MARGIN = 0.08
THIN_MARGIN_RATIO = 0.30
STANDARD_RATIO = 0.70


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def classify_columns(df, exclude=("고유ID", TARGET, "IS_TRAIN")):
    """08번 classify_columns와 동일한 판정 기준."""
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
    t0 = time.time()

    # =====================================================================
    # 1) 07번 원본(파생변수 포함, 인코딩 이전) 로드 - 라벨 있는 행만
    # =====================================================================
    log("=" * 70)
    log("full_feature_table_derived.pkl 로드")
    df = pd.read_pickle(f"{OUT}/full_feature_table_derived.pkl")
    labeled = df[df["IS_TRAIN"] == 1].copy().reset_index(drop=True)
    labeled[TARGET] = labeled[TARGET].astype(int)
    log(f"라벨 있는 데이터: {labeled.shape}, 부도율={labeled[TARGET].mean():.4%}")

    # =====================================================================
    # 2) 컬럼 분류 + 라벨인코딩 (매핑표를 아티팩트로 저장)
    # =====================================================================
    char_cat_cols, binary_flag_cols, numeric_cols = classify_columns(labeled)
    log(f"컬럼 분류: 문자범주형 {len(char_cat_cols)}개 / 이진플래그 {len(binary_flag_cols)}개 "
        f"/ 수치형 {len(numeric_cols)}개")

    label_maps = {}
    for c in char_cat_cols:
        cats = pd.Categorical(labeled[c].fillna("MISSING")).categories
        mapping = {cat: i for i, cat in enumerate(cats)}
        label_maps[c] = mapping
        labeled[c] = labeled[c].fillna("MISSING").map(mapping).astype("int16")

    feat_cols = char_cat_cols + binary_flag_cols + numeric_cols
    X_all = labeled[feat_cols]
    y_all = labeled[TARGET].values
    median_fill = X_all[numeric_cols].median() if numeric_cols else pd.Series(dtype=float)
    X_all = X_all.fillna(median_fill)

    # =====================================================================
    # 3) 85(학습)/15(보정) 분할 후 XGBoost + Platt 보정기 학습
    # =====================================================================
    log("=" * 70)
    log("85(학습)/15(보정)로 분할 후 최종 학습")
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_STATE
    )
    log(f"fit={X_fit.shape}, calib={X_calib.shape}")

    n_pos, n_neg = y_fit.sum(), len(y_fit) - y_fit.sum()
    spw = n_neg / n_pos
    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="auc",
                         random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")
    xgb.fit(X_fit, y_fit)
    log("XGBoost 학습 완료")

    p_calib_raw = xgb.predict_proba(X_calib)[:, 1]
    auc_calib = roc_auc_score(y_calib, p_calib_raw)
    log(f"보정용 15% 기준 AUC(참고용) = {auc_calib:.4f}")

    eps = 1e-6

    def to_logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    platt = LogisticRegression()
    platt.fit(to_logit(p_calib_raw), y_calib)
    p_calib = platt.predict_proba(to_logit(p_calib_raw))[:, 1]

    # =====================================================================
    # 4) 등급구간(18번 방법3) + 의사결정규칙(23번) 재계산 - 전체 데이터 기준
    # =====================================================================
    log("=" * 70)
    log("등급구간 및 의사결정규칙 재계산 중...")
    base_rate = y_all.mean()
    grade_bins = [-0.001, base_rate / 4, base_rate / 2, base_rate, base_rate * 2, 1.001]
    log(f"기저율={base_rate:.4%} 기준 등급구간: "
        f"A<{base_rate/4:.2%}, B<{base_rate/2:.2%}, C<{base_rate:.2%}, "
        f"D<{base_rate*2:.2%}, E>={base_rate*2:.2%}")

    grade_calib = pd.cut(p_calib, bins=grade_bins, labels=GRADES)
    loan_col = "대출금액" if "대출금액" in labeled.columns else None
    decision_df = pd.DataFrame({
        "등급": grade_calib,
        "실제": y_calib,
        "대출금액": X_calib[loan_col].values if loan_col else np.nan,
    })
    grade_stats = (
        decision_df.groupby("등급", observed=False)
        .agg(인원수=("실제", "count"), 실제부도율=("실제", "mean"), 평균대출금액=("대출금액", "mean"))
        .reindex(GRADES)
    )
    p_grade = grade_stats["실제부도율"]
    grade_stats["EV_비율"] = (1 - p_grade) * NET_MARGIN - p_grade * LGD
    grade_stats["EV_금액"] = grade_stats["EV_비율"] * grade_stats["평균대출금액"]
    grade_stats["위험조정가산금리(%p)"] = (LGD * p_grade / (1 - p_grade)) * 100
    ev_a = grade_stats.loc["A", "EV_금액"]
    grade_stats["권장조치"] = [decide(ev, ev_a) for ev in grade_stats["EV_금액"]]

    print(grade_stats.round(4).to_string())
    decision_table = grade_stats.to_dict(orient="index")

    # =====================================================================
    # 5) 아티팩트 번들 저장
    # =====================================================================
    log("=" * 70)
    log("아티팩트 저장 중...")
    artifact = {
        "feat_cols": feat_cols,
        "char_cat_cols": char_cat_cols,
        "binary_flag_cols": binary_flag_cols,
        "numeric_cols": numeric_cols,
        "label_maps": label_maps,
        "median_fill": median_fill,
        "xgb_model": xgb,
        "platt_model": platt,
        "platt_eps": eps,
        "grade_bins": grade_bins,
        "grades": GRADES,
        "base_rate": base_rate,
        "decision_table": decision_table,
        "lgd": LGD,
        "net_margin": NET_MARGIN,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{OUT}/최종_스코어링_아티팩트.pkl", "wb") as f:
        pickle.dump(artifact, f)

    log("저장 완료: 최종_스코어링_아티팩트.pkl")
    log(f"전체 소요시간: {time.time() - t0:.1f}초")
