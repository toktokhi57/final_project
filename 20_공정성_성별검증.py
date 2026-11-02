# -*- coding: utf-8 -*-
"""
STEP 20. 공정성(성별) 검증 - 성별 변수 제거 실험

배경:
- 16번 SHAP 분석에서 성별이 전역 중요도 3위로 나와, 모델이 성별을 강하게 학습하고 있다는
  게 확인됐다. 06번 EDA(카이제곱=920.01, p<0.001)에서도 성별에 따라 실제 부도율이
  유의하게 달랐다(남 10.14% vs 여 7.00%, 06번 기준).
- 신용평가에서 성별을 직접 변수로 쓰는 건 국가/규정에 따라 차별로 간주될 수 있다(미국
  ECOA, EU 성별지침 등). 이 스크립트는 두 가지를 실증적으로 확인한다.
  1) 성별을 빼면 모델 성능(AUC)이 실제로 얼마나 떨어지는가
  2) 성별을 빼도 다른 변수들(프록시)을 통해 성별 간 예측확률 격차가 그대로 남는가
     -> 남는다면 "프록시 차별"이 존재한다는 뜻이라 성별 제거만으로는 공정성 문제가
     해결되지 않는다.

절차:
  1) [모델A] 전체 변수(성별 포함) - 12번과 동일 파라미터로 학습
  2) [모델B] 성별 변수만 제외 - 나머지는 동일 파라미터/동일 데이터로 학습
  3) 두 모델의 valid셋 AUC 비교 -> 성별 제거로 인한 성능 손실 확인
  4) 두 모델 각각에서 "성별 그룹별 평균 예측확률"과 "임계값(0.665) 기준 고위험
     판정 비율(flag rate)"을 비교
  5) 4/5 규칙(disparate impact, 미국 EEOC 관행 기준: 두 그룹 flag rate 비율이
     0.8 미만이면 불균형으로 판정) 적용

입력: tree_encoded_train.pkl, tree_encoded_valid.pkl (08번 출력)
출력: 공정성_성별검증_결과.csv, 공정성_성별검증_비교.png
"""
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42
GENDER_COL = "성별"
FINAL_THRESHOLD = 0.665  # 15/15_1번에서 확정한 임계값


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def train_xgb(X, y):
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    spw = n_neg / n_pos
    model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="auc",
                           random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")
    model.fit(X, y)
    return model


if __name__ == "__main__":
    t0 = time.time()

    # =====================================================================
    # 1) 데이터 로드
    # =====================================================================
    log("=" * 70)
    log("데이터 로드")
    tr = pd.read_pickle(f"{OUT}/tree_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl")
    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols_full = [c for c in tr.columns if c not in exclude]
    if GENDER_COL not in feat_cols_full:
        raise ValueError(f"'{GENDER_COL}' 컬럼을 찾을 수 없습니다. tree_encoded_train.pkl의 "
                          f"실제 성별 컬럼명을 확인하고 스크립트 상단 GENDER_COL을 수정하세요.")
    feat_cols_nogender = [c for c in feat_cols_full if c != GENDER_COL]

    med = tr[feat_cols_full].median(numeric_only=True)
    X_tr_full = tr[feat_cols_full].fillna(med)
    X_val_full = val[feat_cols_full].fillna(med)
    X_tr_nogender = X_tr_full[feat_cols_nogender]
    X_val_nogender = X_val_full[feat_cols_nogender]
    y_tr = tr[TARGET].astype(int).values
    y_val = val[TARGET].astype(int).values
    gender_val = val[GENDER_COL].values  # 인코딩된 값 그대로 (아래 실제부도율로 그룹 라벨 역추정)
    del tr, val

    # =====================================================================
    # 2) 모델A(성별 포함) / 모델B(성별 제외) 학습
    # =====================================================================
    log("=" * 70)
    log("[모델A] 성별 포함 학습 중...")
    model_a = train_xgb(X_tr_full, y_tr)
    p_val_a = model_a.predict_proba(X_val_full)[:, 1]

    log("[모델B] 성별 제외 학습 중...")
    model_b = train_xgb(X_tr_nogender, y_tr)
    p_val_b = model_b.predict_proba(X_val_nogender)[:, 1]

    auc_a = roc_auc_score(y_val, p_val_a)
    auc_b = roc_auc_score(y_val, p_val_b)
    log(f"[모델A-성별포함] AUC={auc_a:.4f}")
    log(f"[모델B-성별제외] AUC={auc_b:.4f}")
    log(f"AUC 차이(A-B) = {auc_a - auc_b:+.4f}  "
        f"({'성별 제외해도 성능 거의 그대로' if abs(auc_a - auc_b) < 0.002 else '성별 제외 시 성능 변화 있음'})")

    # =====================================================================
    # 3) 성별 그룹별 실제부도율(그룹 라벨 확인용) + 예측확률/flag율 비교
    # =====================================================================
    log("=" * 70)
    log("성별 그룹별 실제부도율 (그룹 라벨 확인용 - 06번 EDA 기준: 남 10.14% / 여 7.00%와 대조)")
    groups = np.unique(gender_val)
    rows = []
    for g in groups:
        mask = gender_val == g
        n = int(mask.sum())
        actual_rate = y_val[mask].mean()
        mean_p_a = p_val_a[mask].mean()
        mean_p_b = p_val_b[mask].mean()
        flag_rate_a = (p_val_a[mask] >= FINAL_THRESHOLD).mean()
        flag_rate_b = (p_val_b[mask] >= FINAL_THRESHOLD).mean()
        rows.append({
            "성별_인코딩값": g, "인원수": n, "실제부도율": actual_rate,
            "평균예측확률_모델A(성별포함)": mean_p_a, "평균예측확률_모델B(성별제외)": mean_p_b,
            "고위험판정율_모델A": flag_rate_a, "고위험판정율_모델B": flag_rate_b,
        })
        log(f"  그룹={g}: n={n}, 실제부도율={actual_rate:.4%}, "
            f"평균예측확률(A/B)={mean_p_a:.4f}/{mean_p_b:.4f}, "
            f"고위험판정율(A/B)={flag_rate_a:.4%}/{flag_rate_b:.4%}")

    result_df = pd.DataFrame(rows)

    # =====================================================================
    # 4) 4/5 규칙(disparate impact) 판정
    # =====================================================================
    if len(groups) == 2:
        fr_a = result_df["고위험판정율_모델A"].values
        fr_b = result_df["고위험판정율_모델B"].values
        ratio_a = fr_a.min() / fr_a.max() if fr_a.max() > 0 else np.nan
        ratio_b = fr_b.min() / fr_b.max() if fr_b.max() > 0 else np.nan
        log("=" * 70)
        log(f"[모델A-성별포함] 4/5규칙 비율 = {ratio_a:.3f}  "
            f"({'기준(0.8) 미달 -> 불균형 있음' if ratio_a < 0.8 else '기준(0.8) 충족'})")
        log(f"[모델B-성별제외] 4/5규칙 비율 = {ratio_b:.3f}  "
            f"({'기준(0.8) 미달 -> 불균형 있음' if ratio_b < 0.8 else '기준(0.8) 충족'})")
        log("")
        if ratio_b < 0.8:
            log(">>> 성별을 제거했는데도 4/5규칙을 못 넘긴다면, 다른 변수를 통해 성별이 "
                "간접적으로(프록시) 학습되고 있다는 뜻입니다.")
        else:
            log(">>> 성별 제거만으로 4/5규칙을 충족한다면, 프록시 효과보다는 성별 변수 "
                "자체의 직접적 영향이 컸다는 뜻입니다.")
    else:
        log(f"성별 인코딩값이 {len(groups)}개로 나와 4/5규칙(2그룹 비교용)은 건너뜁니다: {groups}")

    result_df.to_csv(f"{OUT}/공정성_성별검증_결과.csv", index=False, encoding=ENC)

    # =====================================================================
    # 5) 시각화
    # =====================================================================
    log("=" * 70)
    log("시각화 생성 중...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = [str(g) for g in groups]
    x = np.arange(len(groups))
    width = 0.35

    ax = axes[0]
    ax.bar(x, result_df["실제부도율"] * 100, color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("그룹별 실제 부도율(%)")
    ax.set_xlabel("성별_인코딩값")

    ax = axes[1]
    ax.bar(x - width / 2, result_df["평균예측확률_모델A(성별포함)"] * 100, width, label="모델A(성별포함)")
    ax.bar(x + width / 2, result_df["평균예측확률_모델B(성별제외)"] * 100, width, label="모델B(성별제외)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("그룹별 평균 예측확률(%)")
    ax.set_xlabel("성별_인코딩값")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(x - width / 2, result_df["고위험판정율_모델A"] * 100, width, label="모델A(성별포함)")
    ax.bar(x + width / 2, result_df["고위험판정율_모델B"] * 100, width, label="모델B(성별제외)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"그룹별 고위험 판정율(%) (임계값={FINAL_THRESHOLD})")
    ax.set_xlabel("성별_인코딩값")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}/공정성_성별검증_비교.png", dpi=150, bbox_inches="tight")
    plt.close()

    log("=" * 70)
    log("저장 완료: 공정성_성별검증_결과.csv, 공정성_성별검증_비교.png")
    log(f"전체 소요시간: {time.time() - t0:.1f}초")
