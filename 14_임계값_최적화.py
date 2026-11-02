# -*- coding: utf-8 -*-
"""
STEP 13. 임계값(threshold) 최적화
- 목적: 지금까지는 확률 0.5를 기준으로 부도/정상을 나눴는데, 이게 항상 최선은 아니다.
  특히 XGBoost처럼 scale_pos_weight로 불균형을 보정한 모델은 확률 분포 자체가
  한쪽으로 치우쳐 있어서, 0.5가 아니라 데이터에 맞는 판정 기준점을 따로 찾아야
  실질적인 성능(Recall/Precision 균형, 실제 비용)을 제대로 뽑아낼 수 있다.

세 가지 방식으로 최적 임계값을 각각 구하고 비교한다:
  1) Youden's J statistic: ROC 커브에서 TPR-FPR이 최대가 되는 지점.
     민감도(Recall)와 특이도를 동시에 가장 잘 만족하는, 통계적으로 표준적인 기준.
  2) F1 최대화: Precision과 Recall의 조화평균(F1)이 가장 높아지는 지점.
     Precision/Recall을 균형 있게 보고 싶을 때 사용.
  3) 비용 기반 최적화: 신용평가에서 실무적으로 가장 의미 있는 방법.
     - FN(부도자를 "정상"으로 놓침) 비용과 FP(우량고객을 "부도"로 잘못 거절) 비용에
       가중치를 매겨서, 총비용(FN건수*FN비용 + FP건수*FP비용)이 최소가 되는 지점을 찾는다.
     - 비용 비율(FN:FP)은 실제 대손비용 데이터가 없어서 한 값으로 확정하지 않고,
       COST_FN_RATIOS 목록(기본 5~12)을 한 번에 돌려서 비율별로 최적 임계값이 어떻게
       바뀌는지 민감도 분석 형태로 비교한다. 비율이 커질수록(부도자를 더 무겁게 볼수록)
       임계값이 낮아지고 Recall이 올라가는 대신 FP(우량고객 오거절)가 늘어나는 트레이드오프가
       나타나는데, 이 표에서 비율이 하나 오를 때마다 "부도자 1명을 추가로 더 잡는 데 우량고객
       몇 명을 추가로 희생하는지"(한계교환비율)도 같이 계산해서 어디서부터 효율이 급격히
       나빠지는지(수확체감 구간)를 확인할 수 있게 했다.

모델: XGBoost(12/13번에서 최종 후보로 확정). 13번에서 튜닝해도 성능 차이가 없었으므로
      재현이 간단한 08/12번 기본 파라미터를 그대로 사용.

입력: tree_encoded_train/valid.pkl (08번 출력)
출력: 임계값_최적화_결과.csv (기본값/Youden/F1 비교),
      임계값_비용비율_비교.csv (FN:FP 비율별 민감도 분석 + 한계교환비율),
      임계값_최적화_ROC.png, 임계값_비용비율_추이.png
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42

# 비용 기반 최적화 가정치 (실제 대손비용 데이터가 없을 때 업계에서 통상 쓰는 비율)
# 하나로 확정하지 않고 여러 비율을 한 번에 비교해서 민감도 분석을 한다.
COST_FN_RATIOS = [7.0, 7.2, 7.4, 7.6, 7.8, 8.0]  # FN 비용이 FP 비용의 몇 배인지 (7~8 구간 확대)
COST_FP = 1  # 우량고객을 "부도"로 잘못 거절한 비용 (기회손실) - 기준값 1로 고정


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def metrics_at_threshold(y_true, p, threshold, cost_fn=None, cost_fp=COST_FP):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    row = {
        "Threshold": round(float(threshold), 4),
        "Accuracy": round(accuracy_score(y_true, pred), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if cost_fn is not None:
        total_cost = fn * cost_fn + fp * cost_fp
        row["총비용"] = int(total_cost)
        row["건당평균비용"] = round(total_cost / len(y_true), 4)
    return row


if __name__ == "__main__":
    t0 = time.time()

    log("=" * 70)
    log("데이터 로드 및 XGBoost 학습")
    tr = pd.read_pickle(f"{OUT}/tree_encoded_train.pkl")
    val = pd.read_pickle(f"{OUT}/tree_encoded_valid.pkl")
    exclude = {"고유ID", TARGET, "IS_TRAIN"}
    feat_cols = [c for c in tr.columns if c not in exclude]
    med = tr[feat_cols].median(numeric_only=True)
    X_tr = tr[feat_cols].fillna(med)
    X_val = val[feat_cols].fillna(med)
    y_tr = tr[TARGET].astype(int).values
    y_val = val[TARGET].astype(int).values
    del tr, val

    n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    spw = n_neg / n_pos

    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="auc",
                         random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")
    xgb.fit(X_tr, y_tr)
    p_val = xgb.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val)
    log(f"XGBoost 학습 완료. Valid AUC={auc:.4f}")

    # 14_1번(임계값 하나씩 직접 테스트)에서 재학습 없이 재사용할 수 있도록 저장
    np.save(f"{OUT}/xgb_valid_proba.npy", p_val)
    np.save(f"{OUT}/xgb_valid_y.npy", y_val)
    log("확률/정답값 저장 완료 (xgb_valid_proba.npy, xgb_valid_y.npy)")

    # =====================================================================
    # 1) Youden's J statistic
    # =====================================================================
    log("=" * 70)
    log("1) Youden's J statistic 탐색")
    fpr, tpr, roc_thresh = roc_curve(y_val, p_val)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    th_youden = roc_thresh[best_j_idx]
    log(f"Youden's J 최적 임계값 = {th_youden:.4f} (J={j_scores[best_j_idx]:.4f}, "
        f"TPR={tpr[best_j_idx]:.4f}, FPR={fpr[best_j_idx]:.4f})")

    # =====================================================================
    # 2) F1 최대화
    # =====================================================================
    log("=" * 70)
    log("2) F1 최대화 지점 탐색")
    prec, rec, pr_thresh = precision_recall_curve(y_val, p_val)
    # precision_recall_curve는 thresholds 개수가 precision/recall보다 1개 적어서 마지막 값을 뺌
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_f1_idx = np.argmax(f1s)
    th_f1 = pr_thresh[best_f1_idx]
    log(f"F1 최적 임계값 = {th_f1:.4f} (F1={f1s[best_f1_idx]:.4f})")

    # =====================================================================
    # 3) 비용 기반 최적화 - 여러 FN:FP 비율을 한 번에 비교 (민감도 분석)
    # =====================================================================
    log("=" * 70)
    log(f"3) 비용 기반 최적화 탐색 (FN:FP 비율 {COST_FN_RATIOS}을 한 번에 비교)")
    # 비율 간격이 좁을 때(예: 7.0~8.0을 0.2 단위로) 임계값 격자가 너무 성기면 서로 다른
    # 비율이 같은 격자점으로 반올림돼 동률로 나올 수 있어, 격자를 0.001 간격으로 촘촘히 함
    cand_thresholds = np.linspace(0.01, 0.99, 981)

    ratio_rows = []
    ratio_thresholds = {}
    for cost_fn in COST_FN_RATIOS:
        costs = np.array([
            metrics_at_threshold(y_val, p_val, th, cost_fn=cost_fn)["총비용"]
            for th in cand_thresholds
        ])
        best_idx = int(np.argmin(costs))
        th_best = cand_thresholds[best_idx]
        ratio_thresholds[cost_fn] = th_best
        row = {"FN:FP비율": f"{cost_fn}:1", **metrics_at_threshold(y_val, p_val, th_best, cost_fn=cost_fn)}
        ratio_rows.append(row)
        log(f"FN:FP={cost_fn}:1 -> 최적 임계값={th_best:.4f}, Recall={row['Recall']:.4f}, "
            f"Precision={row['Precision']:.4f}, 총비용={row['총비용']}")

    ratio_result = pd.DataFrame(ratio_rows)
    # 한계교환비율: 비율이 하나 커질 때 부도자(TP)를 몇 명 더 잡는 대신
    # 우량고객(FP)을 몇 명 더 희생하는지 (직전 행 대비 증분)
    ratio_result["ΔTP(직전대비)"] = ratio_result["TP"].diff()
    ratio_result["ΔFP(직전대비)"] = ratio_result["FP"].diff()
    ratio_result["FP당_추가포착"] = (ratio_result["ΔFP(직전대비)"] / ratio_result["ΔTP(직전대비)"]).round(2)
    ratio_result.to_csv(f"{OUT}/임계값_비용비율_비교.csv", index=False, encoding=ENC)
    log("=== FN:FP 비율별 최적 임계값 비교 (한계교환비율 포함) ===")
    print(ratio_result.to_string(index=False))

    # 대표값(중간 비율)은 기존 결과 표/ROC 표시용으로 사용
    mid_ratio = COST_FN_RATIOS[len(COST_FN_RATIOS) // 2]
    th_cost = ratio_thresholds[mid_ratio]

    # =====================================================================
    # 결과 정리 (기본값 0.5, Youden's J, F1 최대화 비교표)
    # =====================================================================
    log("=" * 70)
    rows = [
        {"방법": "기본값(0.5)", **metrics_at_threshold(y_val, p_val, 0.5)},
        {"방법": "Youden's J", **metrics_at_threshold(y_val, p_val, th_youden)},
        {"방법": "F1 최대화", **metrics_at_threshold(y_val, p_val, th_f1)},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(f"{OUT}/임계값_최적화_결과.csv", index=False, encoding=ENC)
    log("=== 임계값 최적화 결과 (기본값/Youden/F1) ===")
    print(result.to_string(index=False))

    # =====================================================================
    # 시각화1: ROC 커브 위에 임계값 지점 표시
    # =====================================================================
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    for name, th in [("Youden's J", th_youden), ("F1 최대화", th_f1),
                      (f"비용기반(중간값 {mid_ratio}:1)", th_cost)]:
        idx = int(np.argmin(np.abs(roc_thresh - th)))
        ax.scatter(fpr[idx], tpr[idx], s=80, label=f"{name} (th={th:.3f})", zorder=5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve와 최적 임계값 위치")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/임계값_최적화_ROC.png", dpi=150)
    plt.close()

    # =====================================================================
    # 시각화2: FN:FP 비율에 따른 임계값/Recall/Precision/FP-TP 추이
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = ratio_result["FN:FP비율"]

    ax0 = axes[0]
    ax0.plot(x, ratio_result["Threshold"], marker="o")
    ax0.set_xlabel("FN:FP 비율")
    ax0.set_ylabel("최적 임계값")
    ax0.set_title("비율에 따른 최적 임계값 변화")

    ax1 = axes[1]
    ax1.plot(x, ratio_result["Recall"], marker="o", label="Recall")
    ax1.plot(x, ratio_result["Precision"], marker="o", label="Precision")
    ax1.set_xlabel("FN:FP 비율")
    ax1.set_ylabel("비율")
    ax1.set_title("비율에 따른 Recall/Precision 변화")
    ax1.legend()

    ax2 = axes[2]
    ax2.plot(x[1:], ratio_result["FP당_추가포착"][1:], marker="o", color="darkred")
    ax2.set_xlabel("FN:FP 비율 (직전 대비)")
    ax2.set_ylabel("부도자 1명 추가 포착당 희생되는 우량고객 수")
    ax2.set_title("한계교환비율 (수확체감 확인용)")

    plt.tight_layout()
    plt.savefig(f"{OUT}/임계값_비용비율_추이.png", dpi=150)
    plt.close()

    log(f"전체 소요시간: {time.time()-t0:.1f}초")
