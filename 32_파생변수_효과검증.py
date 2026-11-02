# -*- coding: utf-8 -*-
"""
STEP 32. 파생변수 효과 — 통제 비교

■ 왜 필요한가
  기존 두 CSV(모델비교_..._파생변수 유무)는 파일 생성 시각이 26시간 차이다.
  그 사이 결측치 처리나 데이터 버전이 바뀌었으면 '파생변수 효과'가 아니라
  '데이터가 달라진 효과'를 본 것이 된다.

  이 스크립트는 하나의 프로세스 안에서
    · 같은 입력 파일
    · 같은 split (같은 random_state, 같은 stratify)
    · 같은 하이퍼파라미터
  를 고정하고 피처 목록만 바꿔 두 번 학습한다.
  다른 변수가 낄 여지가 없으므로 차이는 전부 파생변수 때문이다.

■ 출력
  파생변수_효과검증.csv   조건 A(파생 제외) vs B(파생 포함) 지표 비교
  파생변수_효과검증_피처목록.csv   어떤 컬럼을 파생으로 판정했는지 (반드시 눈으로 확인)

■ 주의
  DERIVED 목록이 실제 파생변수와 일치하는지 먼저 확인할 것.
  자동 판정이 틀리면 결과 전체가 무의미하다.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)
import xgboost as xgb

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
SRC = f"{OUT}/full_feature_table_derived.pkl"
TARGET = "목표변수_상환어려움여부"
SEED = 42
TEST_SIZE = 0.2

# ── 파생변수 16개 (실제 컬럼명 확인 완료) ──────────────────────
#    11번 슬라이드의 5개 계열과 1:1 대응한다.
DERIVED_EXACT = [
    # ① 소득 대비 부담 비율 (4)
    "신용부담비율_대출소득", "신용부담비율_상환소득",
    "신용부담비율_상환대출", "계약금비율",
    # ② 근속 · 외부점수 (5)
    "근속연령비율", "외부신용점수_평균", "외부신용점수_최솟값",
    "외부신용점수_표준편차", "외부신용점수_결측개수",
    # ③ 정보 신뢰도 (3)
    "정보불일치_총합", "서류제출_총건수", "최근신용조회_총합",
    # ④ 이력 요약 (3)
    "BUREAU_활성대출비율", "PREV_거절이력여부", "연체이력존재여부",
    # ⑤ 기타 (1)
    "가구원당소득",
]
DERIVED_CONTAINS = []
EXPECT_N = 16
# ══════════════════════════════════════════════════════════════


def log(m=""):
    print(f"[{time.strftime('%H:%M:%S')}] {m}" if m else "")


def is_derived(c):
    if c in DERIVED_EXACT:
        return True
    return any(k in c for k in DERIVED_CONTAINS)


def evaluate(name, model, Xtr, ytr, Xva, yva, scale=False):
    t0 = time.time()
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    model.fit(Xtr, ytr)
    el = time.time() - t0
    out = {}
    for tag, X, y in [("Train", Xtr, ytr), ("Valid", Xva, yva)]:
        p = model.predict_proba(X)[:, 1]
        pred = (p >= 0.5).astype(int)
        out[f"{tag}_AUC"] = round(roc_auc_score(y, p), 4)
        out[f"{tag}_Accuracy"] = round(accuracy_score(y, pred), 4)
        out[f"{tag}_Precision"] = round(precision_score(y, pred, zero_division=0), 4)
        out[f"{tag}_Recall"] = round(recall_score(y, pred, zero_division=0), 4)
        out[f"{tag}_F1"] = round(f1_score(y, pred, zero_division=0), 4)
    out["학습시간(초)"] = round(el, 1)
    log(f"    {name:16s} Valid AUC {out['Valid_AUC']:.4f} · "
        f"Recall {out['Valid_Recall']:.4f} · {el:.1f}초")
    return out


if __name__ == "__main__":
    T0 = time.time()

    df = pd.read_pickle(SRC)
    df = df[df["IS_TRAIN"] == 1].copy()
    df[TARGET] = df[TARGET].astype(int)
    log(f"입력 {len(df):,}행 · {df.shape[1]}컬럼")

    drop_cols = [TARGET, "IS_TRAIN"]
    drop_cols += [c for c in df.columns if "ID" == c[-2:] or c.startswith("SK_ID")]
    feats = [c for c in df.columns if c not in drop_cols]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]

    derived = [c for c in feats if is_derived(c)]
    base = [c for c in feats if c not in derived]

    log(f"전체 피처 {len(feats)}개 = 원본 {len(base)}개 + 파생 {len(derived)}개")
    log("파생으로 판정한 컬럼:")
    for c in derived:
        log(f"    {c}")
    missing = [c for c in DERIVED_EXACT if c not in feats]
    if missing:
        log("")
        log(f"[경고] 데이터에 없는 파생변수 {len(missing)}개: {missing}")
    if len(derived) != EXPECT_N:
        log(f"[경고] 기대 {EXPECT_N}개인데 {len(derived)}개만 잡힘 — 목록 확인 필요")
    if not derived:
        raise SystemExit("파생변수가 하나도 잡히지 않음 — DERIVED 목록을 수정하세요")

    pd.DataFrame({"컬럼": feats,
                  "구분": ["파생" if c in derived else "원본" for c in feats]}
                 ).to_csv(f"{OUT}/파생변수_효과검증_피처목록.csv",
                          index=False, encoding="utf-8-sig")

    # ── split 은 단 한 번만 만든다 (두 조건이 완전히 같은 행을 쓰도록) ──
    y = df[TARGET].values
    idx_tr, idx_va = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED, stratify=y)
    log(f"split 고정 — 학습 {len(idx_tr):,} · 검증 {len(idx_va):,} "
        f"(부도율 {y[idx_tr].mean():.4%} / {y[idx_va].mean():.4%})")

    n_pos, n_neg = int(y[idx_tr].sum()), int((1 - y[idx_tr]).sum())
    spw = n_neg / n_pos

    rows = []
    for cond, cols in [("A_파생제외", base), ("B_파생포함", feats)]:
        log("")
        log(f"[{cond}]  피처 {len(cols)}개")
        X = df[cols].fillna(df[cols].median()).values
        Xtr, Xva = X[idx_tr], X[idx_va]
        ytr, yva = y[idx_tr], y[idx_va]

        # 발표 자료의 실제 설정과 맞춘다.
        #   RandomForest : class_weight='balanced'  (15번 슬라이드 채택안)
        #   XGBoost      : scale_pos_weight        (최종 모델 설정)
        #   로지스틱      : 가중치 없음             (원본 비교표와 동일)
        models = [
            ("로지스틱회귀(표준화)",
             LogisticRegression(max_iter=1000, random_state=SEED), True),
            ("RandomForest",
             RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1), False),
            ("XGBoost",
             xgb.XGBClassifier(n_estimators=500, max_depth=5,
                               learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, scale_pos_weight=spw,
                               eval_metric="auc", random_state=SEED,
                               n_jobs=-1, tree_method="hist"), False),
        ]
        try:
            import lightgbm as lgbm
            models.append(("LightGBM",
                           lgbm.LGBMClassifier(n_estimators=500, max_depth=5,
                                               learning_rate=0.05,
                                               subsample=0.8,
                                               colsample_bytree=0.8,
                                               class_weight="balanced",
                                               random_state=SEED, n_jobs=-1,
                                               verbose=-1), False))
        except ImportError:
            log("    (LightGBM 미설치 → 건너뜀)")
        for name, mdl, sc in models:
            r = evaluate(name, mdl, Xtr, ytr, Xva, yva, sc)
            r.update({"조건": cond, "모델": name, "피처수": len(cols)})
            rows.append(r)

    res = pd.DataFrame(rows)
    order = ["조건", "모델", "피처수", "Valid_AUC", "Valid_Recall",
             "Valid_Precision", "Valid_F1", "Valid_Accuracy",
             "Train_AUC", "Train_Recall", "학습시간(초)"]
    res = res[order]
    res.to_csv(f"{OUT}/파생변수_효과검증.csv", index=False, encoding="utf-8-sig")

    log("")
    log("=" * 70)
    log("파생변수 순효과 (B − A)")
    log("=" * 70)
    a = res[res["조건"] == "A_파생제외"].set_index("모델")
    b = res[res["조건"] == "B_파생포함"].set_index("모델")
    for m in a.index:
        d_auc = b.loc[m, "Valid_AUC"] - a.loc[m, "Valid_AUC"]
        d_rec = b.loc[m, "Valid_Recall"] - a.loc[m, "Valid_Recall"]
        log(f"  {m:20s} AUC {a.loc[m,'Valid_AUC']:.4f} → "
            f"{b.loc[m,'Valid_AUC']:.4f} ({d_auc:+.4f})   "
            f"Recall {a.loc[m,'Valid_Recall']:.4f} → "
            f"{b.loc[m,'Valid_Recall']:.4f} ({d_rec:+.4f})")

    log("")
    log("저장: 파생변수_효과검증.csv / 파생변수_효과검증_피처목록.csv")
    log(f"소요시간 {time.time()-T0:.1f}초")
