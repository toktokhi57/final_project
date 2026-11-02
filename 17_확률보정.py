# -*- coding: utf-8 -*-
"""
STEP 17. 확률 보정 (Probability Calibration)

배경 (왜 필요한가):
- XGBoost는 scale_pos_weight로 클래스 불균형을 보정해서 학습했다. 이 방식은
  AUC/Recall 등 "순위·판별력"에는 도움이 되지만, predict_proba가 반환하는 값 자체는
  더 이상 진짜 확률이 아니게 된다.
  실제로 16번 SHAP 분석 당시 valid셋 평균 예측확률이 약 51%였는데, 실제 부도율(기저율)은
  8.07%에 불과했다 - 약 6배 과대추정.
- 지금까지 임계값(0.665)은 "이 값 이상이면 위험군"이라는 순위 기반 의사결정에는 문제가
  없었지만, 다음 단계인 A~E 등급 변환에서는 확률값 자체("이 사람은 실제로 15% 확률로
  부도난다")가 의미를 가져야 하므로 이 왜곡을 반드시 바로잡아야 한다.

방법:
- Platt Scaling (method="sigmoid" 방식): 원본 확률의 로짓(logit)에 로지스틱회귀를 한 번
  더 씌워 보정. 데이터가 적어도 안정적이지만, 보정 곡선이 S자가 아니면 잘 안 맞을 수 있음.
- Isotonic Regression: 단조증가라는 제약만 두고 자유롭게 보정 곡선을 학습. 데이터가
  충분히 많을 때(우리처럼 수십만 행) 더 정교하게 맞출 수 있지만, 과적합 위험이 있어
  반드시 학습에 쓰지 않은 별도 데이터로 검증해야 한다.
- 두 방법 모두 "모델 재학습"이 아니라 "이미 학습된 모델의 출력값에 사후 변환을 씌우는 것"
  이므로, 두 확률 사이의 상대적 순위(=AUC)는 이론상 거의 그대로 유지된다. 바뀌는 것은
  각 확률값이 실제 부도율에 얼마나 가깝게 맞춰지느냐이다.

절차:
  1) train셋을 80/20으로 나눠, 80%(train_fit)로 XGBoost를 학습하고 20%(train_calib)는
     보정기(calibrator) 학습 전용으로 따로 뗀다.
     (valid셋은 지금까지처럼 최종 평가 전용으로 끝까지 손대지 않는다 - 데이터 누수 방지.
     보정기를 valid셋으로 학습하면 신뢰도 곡선이 실제보다 좋아 보이는 착시가 생긴다.)
  2) train_calib에서 나온 원본 예측확률로 Platt/Isotonic 두 보정기를 각각 학습
  3) valid셋에서 원본 vs Platt 보정 vs Isotonic 보정, 세 가지를 비교
     - 신뢰도 곡선(calibration curve, reliability diagram)
     - Brier score (낮을수록 좋음 - 예측확률과 실제 결과(0/1) 간의 평균제곱오차)
     - 평균예측확률 vs 실제 부도율 (이상적으로는 거의 같아야 함)
     - AUC (보정 전후로 거의 변하지 않아야 정상 - 순위는 그대로 유지되므로)
  4) 두 보정기와 보정된 확률을 저장 -> 다음 단계(A~E 등급 변환)에서 그대로 재사용

주의: 이 스크립트는 XGBoost를 train 전체(100%)가 아니라 train_fit(80%)으로 다시 학습하므로,
      여기서 나오는 원본 예측확률(p_val_raw)은 14~16번에서 쓰던 값과 완전히 동일하지는
      않다(훈련 데이터가 20% 줄었으므로 근소한 차이 있음). A~E 등급 구간은 이번 단계에서
      나온 "보정된" 확률을 기준으로 새로 정의할 예정이므로 문제가 되지 않는다.

입력: tree_encoded_train.pkl, tree_encoded_valid.pkl (08번 출력)
출력: 확률보정_비교_결과.csv, 확률보정_신뢰도곡선.png,
      calibrator_platt.pkl, calibrator_isotonic.pkl,
      calibrated_valid_proba_platt.npy, calibrated_valid_proba_isotonic.npy,
      valid_proba_raw.npy, valid_y.npy
"""
import time
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 폰트 + 유니코드 마이너스 기호 처리 (16번과 동일한 방식)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42
CALIB_SIZE = 0.2  # train 중 보정기 학습용으로 떼어낼 비율


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


if __name__ == "__main__":
    t0 = time.time()

    # =====================================================================
    # 1) 데이터 로드 및 train 분할 (train_fit 80% / train_calib 20%)
    # =====================================================================
    log("=" * 70)
    log("데이터 로드")
    tr = pd.read_pickle(f"{OUT}/tree_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl")
    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols = [c for c in tr.columns if c not in exclude]
    med = tr[feat_cols].median(numeric_only=True)

    X_tr_full = tr[feat_cols].fillna(med)
    y_tr_full = tr[TARGET].astype(int).values
    X_val = val[feat_cols].fillna(med)
    y_val = val[TARGET].astype(int).values
    del tr, val

    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_tr_full, y_tr_full, test_size=CALIB_SIZE, stratify=y_tr_full,
        random_state=RANDOM_STATE
    )
    log(f"train_fit={X_fit.shape}, train_calib={X_calib.shape}, valid={X_val.shape}")

    # =====================================================================
    # 2) XGBoost 학습 (12~16번과 동일 파라미터, train_fit만 사용)
    # =====================================================================
    log("=" * 70)
    log("XGBoost 학습 중 (train_fit)...")
    n_pos, n_neg = y_fit.sum(), len(y_fit) - y_fit.sum()
    spw = n_neg / n_pos
    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="auc",
                         random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")
    xgb.fit(X_fit, y_fit)
    log("XGBoost 학습 완료")

    p_calib_raw = xgb.predict_proba(X_calib)[:, 1]
    p_val_raw = xgb.predict_proba(X_val)[:, 1]

    # =====================================================================
    # 3) 보정기 학습 (train_calib 전용 - valid셋은 절대 쓰지 않음)
    # =====================================================================
    log("=" * 70)
    log("보정기(Platt / Isotonic) 학습 중...")

    # Platt scaling: 원본확률의 로짓을 입력으로 하는 1차원 로지스틱회귀
    eps = 1e-6

    def to_logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    platt = LogisticRegression()
    platt.fit(to_logit(p_calib_raw), y_calib)

    def platt_transform(p):
        return platt.predict_proba(to_logit(p))[:, 1]

    # Isotonic regression: 단조증가 제약만으로 원본확률 -> 실제확률 매핑을 직접 학습
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_calib_raw, y_calib)

    p_val_platt = platt_transform(p_val_raw)
    p_val_isotonic = isotonic.predict(p_val_raw)

    # =====================================================================
    # 4) valid셋 기준 원본 vs Platt vs Isotonic 비교
    # =====================================================================
    log("=" * 70)
    log("valid셋 기준 보정 전후 비교")

    actual_rate = y_val.mean()
    results = []
    for name, p in [("원본(보정전)", p_val_raw), ("Platt", p_val_platt), ("Isotonic", p_val_isotonic)]:
        auc = roc_auc_score(y_val, p)
        brier = brier_score_loss(y_val, p)
        mean_p = p.mean()
        results.append({
            "방법": name,
            "AUC": auc,
            "Brier_Score(낮을수록좋음)": brier,
            "평균예측확률": mean_p,
            "실제부도율": actual_rate,
            "괴리(평균예측-실제)": mean_p - actual_rate,
        })
        log(f"[{name}] AUC={auc:.4f}, Brier={brier:.4f}, 평균예측확률={mean_p:.4f} "
            f"(실제부도율={actual_rate:.4f}, 괴리={mean_p - actual_rate:+.4f})")

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{OUT}/확률보정_비교_결과.csv", index=False, encoding=ENC)

    # =====================================================================
    # 5) 신뢰도 곡선 (calibration curve, reliability diagram)
    # =====================================================================
    log("=" * 70)
    log("신뢰도 곡선 생성 중...")
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k--", label="완벽한 보정(이상선)")
    for name, p, style in [("원본(보정전)", p_val_raw, "o-"),
                            ("Platt", p_val_platt, "s-"),
                            ("Isotonic", p_val_isotonic, "^-")]:
        frac_pos, mean_pred = calibration_curve(y_val, p, n_bins=15, strategy="quantile")
        plt.plot(mean_pred, frac_pos, style, label=name)
    plt.xlabel("예측확률 (구간 평균)")
    plt.ylabel("실제 부도율 (구간 내)")
    plt.title("신뢰도 곡선 (Calibration Curve) - valid셋 기준")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/확률보정_신뢰도곡선.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =====================================================================
    # 6) 보정기 및 보정된 확률 저장 (다음 단계=A~E 등급 변환에서 재사용)
    # =====================================================================
    log("=" * 70)
    log("보정기 및 결과 저장 중...")
    with open(f"{OUT}/calibrator_platt.pkl", "wb") as f:
        pickle.dump({"model": platt, "eps": eps}, f)
    with open(f"{OUT}/calibrator_isotonic.pkl", "wb") as f:
        pickle.dump(isotonic, f)

    np.save(f"{OUT}/calibrated_valid_proba_platt.npy", p_val_platt)
    np.save(f"{OUT}/calibrated_valid_proba_isotonic.npy", p_val_isotonic)
    np.save(f"{OUT}/valid_proba_raw.npy", p_val_raw)
    np.save(f"{OUT}/valid_y.npy", y_val)

    log("저장 완료: 확률보정_비교_결과.csv, 확률보정_신뢰도곡선.png, "
        "calibrator_platt.pkl, calibrator_isotonic.pkl, calibrated_valid_proba_*.npy")
    log("=" * 70)
    log(f"전체 소요시간: {time.time() - t0:.1f}초")
    log("")
    log("다음 단계 안내: Brier Score가 더 낮고 신뢰도 곡선이 이상선(대각선)에 더 가까운 쪽을")
    log("최종 채택해서 A~E 등급 구간을 정의하면 됩니다. (데이터가 충분히 많으면 보통 Isotonic이")
    log("더 유연하게 잘 맞는 경우가 많지만, 실제 결과를 보고 판단하는 게 정확합니다.)")
