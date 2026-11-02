# -*- coding: utf-8 -*-
"""
STEP 10. 불균형 처리 비교
- 목적: 소수클래스(부도, 전체의 약 8%)에 대한 탐지력(Recall) 개선. 지금까지의 RF/XGB/LGBM은
  전부 class_weight="balanced"/scale_pos_weight로 이미 어느 정도 불균형을 보정하고 있었는데,
  "그 보정이 다른 방법들과 비교했을 때 실제로 얼마나/어떻게 효과가 있는지"를 같은 모델
  (RandomForest)로 고정해두고 여러 기법을 갈아끼우면서 직접 비교한다.

  - 알고리즘 레벨 : class_weight="balanced" (RandomForest/LogisticRegression/SVM),
    scale_pos_weight (XGBoost/LightGBM) - 데이터는 그대로 두고 손실함수에서 소수클래스
    오류에 더 큰 페널티를 줌
  - 리샘플링(오버샘플링) : RandomOverSampler(소수클래스를 단순 복제),
    SMOTE(소수클래스 근처를 보간해 새 가상 샘플 생성),
    ADASYN(SMOTE와 비슷하되 "구분이 어려운 경계 근처"를 더 많이 생성)
  - 리샘플링(언더샘플링) : RandomUnderSampler(다수클래스를 무작위로 줄임)
  - 리샘플링(오버+언더 결합) : SMOTETomek(SMOTE로 늘린 뒤 Tomek Links로 경계의 애매한
    다수클래스 샘플을 제거해 경계를 더 뚜렷하게 만듦)

  주의: 리샘플링은 반드시 train에만 적용하고 valid/test는 원본 그대로 둬야 한다.
  (valid까지 리샘플링하면 실제 운영 환경과 다른 분포를 "정답인 것처럼" 평가하게 되어
   결과가 부풀려짐 - 데이터 누수의 한 형태)

  주의2: 이 스크립트는 tree_encoded 데이터(라벨인코딩된 범주형 + 원본 수치형)를 쓰는데,
  SMOTE/ADASYN은 "두 샘플 사이를 선형보간"해서 새 샘플을 만들기 때문에, 라벨인코딩된
  범주형 컬럼(예: 최종학력=0~4)에 적용하면 2.37 같은 존재하지 않는 값이 생길 수 있다.
  이건 SMOTE 계열 오버샘플링의 잘 알려진 한계이며, 범주형이 섞인 데이터에는 원래
  SMOTENC(Nominal+Continuous 버전)를 쓰는 게 정석이다. 여기서는 사용자가 요청한 기법
  그대로(SMOTE/ADASYN) 비교하되, 이 한계를 감안해서 해석해야 한다.

- 비교 방법: RandomForest(하이퍼파라미터 고정)에 아래 7가지 처리를 각각 적용해서
  Train/Valid AUC·Recall·Precision·F1과 학습 소요시간을 한 표로 비교한다.
    1) 처리없음(원본)          : class_weight=None, 리샘플링 없음 (진짜 "처리 전" 기준)
    2) class_weight=balanced  : 알고리즘 레벨 보정만
    3) RandomOverSampler      : 소수클래스 단순 복제로 다수클래스와 개수를 맞춤
    4) RandomUnderSampler     : 다수클래스를 소수클래스 개수로 줄임
    5) SMOTE                  : 소수클래스 근처 보간으로 새 샘플 생성
    6) ADASYN                 : SMOTE의 개선판(경계 근처를 더 많이 생성)
    7) SMOTETomek             : SMOTE + Tomek Links 언더샘플링 결합

- 입력: 08_인코딩_스케일링.py 가 만든 tree_encoded_train/valid.pkl
- 출력: 모델비교_불균형처리_RandomForest.csv, 불균형처리_AUC_Recall_비교.png

필요 패키지: imbalanced-learn (없으면 아래 명령으로 설치)
  conda install -c conda-forge imbalanced-learn
  또는 pip install imbalanced-learn
"""
import time
import gc
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
except ImportError as e:
    raise ImportError(
        "imbalanced-learn 패키지가 필요합니다. Spyder 콘솔에서 "
        "'!pip install imbalanced-learn' 또는 터미널에서 "
        "'conda install -c conda-forge imbalanced-learn' 실행 후 다시 시도하세요."
    ) from e

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


def make_rf(class_weight=None):
    # n_jobs=1 고정: macOS에서 n_jobs=-1이 TimeoutError: [Errno 60]을 유발하는 문제 회피
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=50,
        class_weight=class_weight, n_jobs=1, random_state=RANDOM_STATE,
    )


def run_variant(name, X_tr, y_tr, X_val, y_val, class_weight=None):
    t1 = time.time()
    model = make_rf(class_weight=class_weight)
    model.fit(X_tr, y_tr)
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t1
    log(f"[{name}] train n={len(y_tr)}(부도율={y_tr.mean():.3f}) -> "
        f"Valid AUC={evaluate(y_val, p_val)['AUC']:.4f}, "
        f"Recall={evaluate(y_val, p_val)['Recall']:.4f} ({elapsed:.1f}초)")
    return {
        "Train": evaluate(y_tr, p_tr), "Valid": evaluate(y_val, p_val),
        "학습표본수": len(y_tr), "학습표본_부도율": round(float(y_tr.mean()), 4),
        "학습시간(초)": round(elapsed, 1),
    }


if __name__ == "__main__":
    t0 = time.time()

    tr = pd.read_pickle(f"{OUT}/tree_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl")
    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols = [c for c in tr.columns if c not in exclude]

    med = tr[feat_cols].median(numeric_only=True)
    X_tr = tr[feat_cols].fillna(med).values  # RandomForest는 결측 허용 안 함
    X_val = val[feat_cols].fillna(med).values
    y_tr = tr[TARGET].astype(int).values
    y_val = val[TARGET].astype(int).values
    del tr, val
    gc.collect()
    log(f"로드 완료: train {X_tr.shape}, valid {X_val.shape}, 원본 부도율={y_tr.mean():.4f}")

    results = {}

    # 1) 처리없음(원본) - 진짜 "처리 전" 기준선
    log("=" * 70)
    log("1/7 처리없음(원본)")
    results["처리없음(원본)"] = run_variant("처리없음(원본)", X_tr, y_tr, X_val, y_val, class_weight=None)

    # 2) class_weight=balanced (알고리즘 레벨)
    log("=" * 70)
    log("2/7 class_weight=balanced")
    results["class_weight=balanced"] = run_variant(
        "class_weight=balanced", X_tr, y_tr, X_val, y_val, class_weight="balanced")

    # 3) RandomOverSampler
    log("=" * 70)
    log("3/7 RandomOverSampler")
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_ros, y_ros = ros.fit_resample(X_tr, y_tr)
    results["RandomOverSampler"] = run_variant("RandomOverSampler", X_ros, y_ros, X_val, y_val, class_weight=None)
    del X_ros, y_ros
    gc.collect()

    # 4) RandomUnderSampler
    log("=" * 70)
    log("4/7 RandomUnderSampler")
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_rus, y_rus = rus.fit_resample(X_tr, y_tr)
    results["RandomUnderSampler"] = run_variant("RandomUnderSampler", X_rus, y_rus, X_val, y_val, class_weight=None)
    del X_rus, y_rus
    gc.collect()

    # 5) SMOTE
    log("=" * 70)
    log("5/7 SMOTE (표본 수가 많아 다소 시간이 걸릴 수 있음)")
    smote = SMOTE(random_state=RANDOM_STATE)  # 최신 imbalanced-learn은 n_jobs 파라미터가 없어짐
    X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
    results["SMOTE"] = run_variant("SMOTE", X_sm, y_sm, X_val, y_val, class_weight=None)
    del X_sm, y_sm
    gc.collect()

    # 6) ADASYN
    log("=" * 70)
    log("6/7 ADASYN")
    adasyn = ADASYN(random_state=RANDOM_STATE)
    X_ada, y_ada = adasyn.fit_resample(X_tr, y_tr)
    results["ADASYN"] = run_variant("ADASYN", X_ada, y_ada, X_val, y_val, class_weight=None)
    del X_ada, y_ada
    gc.collect()

    # 7) SMOTETomek (오버+언더 결합)
    log("=" * 70)
    log("7/7 SMOTETomek")
    smt = SMOTETomek(random_state=RANDOM_STATE)
    X_smt, y_smt = smt.fit_resample(X_tr, y_tr)
    results["SMOTETomek"] = run_variant("SMOTETomek", X_smt, y_smt, X_val, y_val, class_weight=None)
    del X_smt, y_smt
    gc.collect()

    # ---- 결과 정리 ----
    log("=" * 70)
    rows = []
    for name, r in results.items():
        row = {"기법": name, "학습표본수": r["학습표본수"], "학습표본_부도율": r["학습표본_부도율"]}
        for split in ("Train", "Valid"):
            for metric, v in r[split].items():
                row[f"{split}_{metric}"] = round(v, 4)
        row["학습시간(초)"] = r["학습시간(초)"]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(f"{OUT}/모델비교_불균형처리_RandomForest.csv", index=False, encoding=ENC)
    log("=== 불균형 처리 비교 결과 (RandomForest 고정) ===")
    print(comparison.to_string(index=False))

    # AUC/Recall 막대그래프
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(comparison["기법"], comparison["Valid_AUC"])
    axes[0].set_title("기법별 Valid AUC")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].axhline(comparison.loc[comparison["기법"] == "처리없음(원본)", "Valid_AUC"].values[0],
                     color="red", linestyle="--", linewidth=1, label="처리없음(원본) 기준선")
    axes[0].legend(fontsize=8)

    axes[1].bar(comparison["기법"], comparison["Valid_Recall"])
    axes[1].set_title("기법별 Valid Recall")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].axhline(comparison.loc[comparison["기법"] == "처리없음(원본)", "Valid_Recall"].values[0],
                     color="red", linestyle="--", linewidth=1, label="처리없음(원본) 기준선")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}/불균형처리_AUC_Recall_비교.png", dpi=150)
    plt.close()

    log(f"전체 소요시간: {time.time()-t0:.1f}초")
