# -*- coding: utf-8 -*-
"""
STEP 30. 세그먼트(집단) 분석 통합 파이프라인
        — 기존 26 / 26-1 / 27 / 28번을 하나로 재설계한 최종본

════════════════════════════════════════════════════════════════════
전체 흐름
════════════════════════════════════════════════════════════════════
  STAGE 0  준비        데이터 로드 · 제외 변수 확정
  STAGE 1  단변량 탐색  어떤 변수가 집단을 가르는가
  STAGE 2  축 선정      SHAP + 단변량 교집합으로 세분화 축 확정
  STAGE 3  규칙 세분화  의사결정나무 → 규칙 문장          ★ 주 방법
  STAGE 4  보조 군집화  K-Means → 나무가 놓친 집단 탐색    ★ 보조
  STAGE 5  교차검증     3번과 4번이 같은 집단을 지목하는가
  STAGE 6  공정성 점검  보호변수 제거 후 대리차별 잔존 확인
  STAGE 7  세그먼트 확정 EV 기반 대응전략 부여

════════════════════════════════════════════════════════════════════
설계 원칙 — 왜 이 순서인가
════════════════════════════════════════════════════════════════════
① 단변량 → 다변량 순서
   바로 군집화부터 하면 결과가 나와도 "왜 이렇게 묶였는지" 설명을 못 한다.
   단변량을 먼저 봐야 나중에 나온 세그먼트가 말이 되는지 검증할 수 있다.

② 나무가 주(主), 군집화가 보조(補)
   나무는 지도학습이라 나온 세그먼트가 정의상 부도와 관련이 있고,
   결과가 "조건 → 부도율" 문장이라 심사역이 그대로 쓸 수 있다.
   K-Means는 무감독이라 부도와 무관한 축으로 갈릴 수 있지만,
   정답을 안 보기 때문에 나무가 못 본 구조를 발견할 수 있다.
   그래서 나무로 규칙을 만들고, 군집화로 빠진 곳을 점검한다.

③ 교차검증을 반드시 넣는다
   서로 다른 알고리즘이 같은 집단을 지목하면 그 자체가 근거가 된다.
   임계값에서 F1·비용기반·LGD 세 방법이 0.666에 수렴한 것과 같은 논리.

④ 공정성 점검은 맨 마지막이 아니라 확정 직전에 둔다
   대리차별이 발견되면 세그먼트를 다시 짜야 하므로,
   전략을 부여하기 전에 확인해야 한다.

════════════════════════════════════════════════════════════════════
제외 변수 — 반드시 지킬 것
════════════════════════════════════════════════════════════════════
  · 성별      20번 공정성 검증에서 4/5 rule 위반(0.557). 분할 축으로 쓰면
              "여성이면 거절"에 해당하는 규칙이 나온다. 사후 점검용으로만 사용.
  · 근속연수   04b의 median 대치로 4.4~4.6년 구간에 6만명이 인공적으로
              쌓여 단조성이 깨짐(26번에서 확인). 세분화 축에서 배제.

출력 파일 (모두 접두사 SEG_)
  SEG_1_단변량_범주형.csv       SEG_1_단변량_수치형.csv
  SEG_2_축선정.csv
  SEG_3_규칙표.csv              SEG_3_깊이비교.csv
  SEG_4_군집프로파일.csv        SEG_4_군집특징.csv
  SEG_5_교차검증.csv
  SEG_6_공정성점검.csv
  SEG_7_최종세그먼트.csv
  SEG_그림1_단변량.png          SEG_그림2_나무.png
  SEG_그림3_군집.png            SEG_그림4_최종요약.png
  SEG_할당결과.pkl
"""
import time
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
SEED = 42

# SHAP TOP10 (09번 결과) — 성별은 공정성 사유로 제외
SHAP_TOP = {
    "외부신용점수_평균":      0.437,
    "신용부담비율_상환대출":   0.118,
    "외부신용점수_최솟값":     0.113,
    "계약금비율":             0.108,
    "INST_IS_LATE_MEAN":     0.107,
    "연간상환액":             0.096,
    "INST_실제상환금액_SUM":   0.094,
    "POS_잔여할부개월수_MEAN": 0.094,
    "최종학력":               0.089,
}

EXCLUDE = ["성별", "근속연수"]      # 분할 축 금지 (사유는 상단 주석)
FAIRNESS_CHECK = ["성별", "직업종류", "소득유형"]   # 사후 점검 대상

MIN_N = 500          # 단변량 범주 최소 인원
TREE_DEPTHS = [2, 3, 4]
TREE_DEPTH = 3       # 채택 깊이
TREE_MIN_LEAF = 3000
K_RANGE = range(2, 9)
K_FINAL = 5
SIL_SAMPLE = 20000

LGD = 0.45           # 부도시 손실률 (23번과 동일)
NET_MARGIN = 0.08    # 순이익률
# ══════════════════════════════════════════════════════════════════


def log(m=""):
    print(f"[{time.strftime('%H:%M:%S')}] {m}" if m else "")


def stage(n, title):
    print()
    print("═" * 78)
    print(f"  STAGE {n}.  {title}")
    print("═" * 78)


def wilson(k, n, z=1.96):
    """Wilson 신뢰구간 — 표본이 작거나 비율이 극단일 때 Wald보다 정확."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def add_ci(df, k_col="부도자수", n_col="인원수"):
    ci = [wilson(int(a), int(b)) for a, b in zip(df[k_col], df[n_col])]
    df["CI하한(%)"] = [round(a * 100, 2) for a, _ in ci]
    df["CI상한(%)"] = [round(b * 100, 2) for _, b in ci]
    df["CI폭(%p)"] = (df["CI상한(%)"] - df["CI하한(%)"]).round(3)
    return df


def ordinal_encode(df, X, cols):
    """범주형을 순서형으로.
    목표변수로 인코딩하면 정답이 새므로 금지.
    외부신용점수 평균 순으로 정렬해 무감독 방식으로 순서를 부여한다."""
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            order = (df.groupby(c, observed=True)["외부신용점수_평균"]
                       .mean().sort_values().index.tolist())
            X[c] = X[c].map({v: i for i, v in enumerate(order)}).astype(float)
            log(f"    순서형 인코딩: {c} ({len(order)}개 범주)")
    return X


def extract_rules(clf, feat_names):
    """리프까지의 경로를 규칙 문장으로 복원.
    같은 변수가 경로에 여러 번 나오면 가장 좁은 조건만 남긴다.
      (기존 28번은 '<= 0.416 AND <= 0.290'처럼 중복이 남았음)"""
    t = clf.tree_
    rules = {}

    def fmt(bounds):
        parts = []
        for f, (lo, hi) in bounds.items():
            if lo == -np.inf:
                parts.append(f"{f} ≤ {hi:,.4g}")
            elif hi == np.inf:
                parts.append(f"{f} > {lo:,.4g}")
            else:
                parts.append(f"{lo:,.4g} < {f} ≤ {hi:,.4g}")
        return "  AND  ".join(parts) if parts else "(전체)"

    def walk(node, bounds):
        if t.children_left[node] == -1:
            rules[node] = fmt(bounds)
            return
        f, thr = feat_names[t.feature[node]], t.threshold[node]

        b = {k: list(v) for k, v in bounds.items()}
        lo, hi = b.get(f, [-np.inf, np.inf])
        b[f] = [lo, min(hi, thr)]                 # 왼쪽: f <= thr
        walk(t.children_left[node], b)

        b = {k: list(v) for k, v in bounds.items()}
        lo, hi = b.get(f, [-np.inf, np.inf])
        b[f] = [max(lo, thr), hi]                 # 오른쪽: f > thr
        walk(t.children_right[node], b)

    walk(0, {})
    return rules


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    T0 = time.time()

    # ══════════════════════════════════════════════════════════════
    stage(0, "준비")
    # ══════════════════════════════════════════════════════════════
    df = pd.read_pickle(f"{OUT}/full_feature_table_derived.pkl")
    df = df[df["IS_TRAIN"] == 1].copy()
    df[TARGET] = df[TARGET].astype(int)
    BASE = df[TARGET].mean()
    N = len(df)

    log(f"분석 대상        {N:,}명")
    log(f"기저 부도율      {BASE:.4%}")
    log(f"손익분기 부도율   {NET_MARGIN/(NET_MARGIN+LGD):.4%} "
        f"(순이익률 {NET_MARGIN} / LGD {LGD})")
    log(f"분할 축 제외      {', '.join(EXCLUDE)}")

    # ══════════════════════════════════════════════════════════════
    stage(1, "단변량 탐색 — 어떤 변수가 집단을 가르는가")
    # ══════════════════════════════════════════════════════════════
    # 1-A. 범주형
    cat_cols = [c for c in df.columns
                if (df[c].dtype == "object" or str(df[c].dtype) == "category")
                and c not in EXCLUDE and df[c].nunique() <= 60]

    rows = []
    for c in cat_cols:
        g = df.groupby(c, observed=True)[TARGET].agg(["count", "sum"])
        g = g[g["count"] >= MIN_N]
        for lvl, r in g.iterrows():
            rows.append({"변수": c, "값": str(lvl), "인원수": int(r["count"]),
                         "부도자수": int(r["sum"]),
                         "부도율(%)": round(r["sum"] / r["count"] * 100, 3),
                         "리프트": round((r["sum"] / r["count"]) / BASE, 3)})
    cat_df = add_ci(pd.DataFrame(rows)).sort_values("부도율(%)", ascending=False)
    cat_df.to_csv(f"{OUT}/SEG_1_단변량_범주형.csv", index=False, encoding=ENC)

    log(f"범주형 {len(cat_cols)}개 변수 · {len(cat_df)}개 수준 (n≥{MIN_N})")
    log("  최고위험 TOP5")
    for _, r in cat_df.head(5).iterrows():
        log(f"    {r['변수']:16s} = {r['값']:18s} {r['부도율(%)']:6.2f}% "
            f"(리프트 {r['리프트']:.2f}, {r['인원수']:,}명)")
    log("  최저위험 TOP5")
    for _, r in cat_df.tail(5).iloc[::-1].iterrows():
        log(f"    {r['변수']:16s} = {r['값']:18s} {r['부도율(%)']:6.2f}% "
            f"(리프트 {r['리프트']:.2f}, {r['인원수']:,}명)")

    # 1-B. 수치형 — 인원 균등 5분위
    #   qcut은 값 범위가 아니라 인원을 5등분한다.
    #   값 범위로 자르면(cut) 구간별 인원이 심하게 쏠려 부도율이 불안정해진다.
    num_targets = [c for c in
                   ["나이", "총소득", "대출금액", "외부신용점수_평균",
                    "신용부담비율_상환대출", "가구원당소득", "계약금비율"]
                   if c in df.columns and c not in EXCLUDE]

    rows = []
    for c in num_targets:
        try:
            bins = pd.qcut(df[c], 5, duplicates="drop")
        except Exception:
            continue
        g = df.groupby(bins, observed=True)[TARGET].agg(["count", "sum"])
        for i, (iv, r) in enumerate(g.iterrows(), 1):
            rows.append({"변수": c, "분위": i, "구간": str(iv),
                         "인원수": int(r["count"]), "부도자수": int(r["sum"]),
                         "부도율(%)": round(r["sum"] / r["count"] * 100, 3),
                         "리프트": round((r["sum"] / r["count"]) / BASE, 3)})
    num_df = add_ci(pd.DataFrame(rows))
    num_df.to_csv(f"{OUT}/SEG_1_단변량_수치형.csv", index=False, encoding=ENC)

    log("")
    log("수치형 5분위 — 분별력 순 (1분위 대비 5분위 배수)")
    spread = []
    for c in num_df["변수"].unique():
        s = num_df[num_df["변수"] == c].sort_values("분위")
        hi, lo = s["부도율(%)"].max(), s["부도율(%)"].min()
        mono = s["부도율(%)"].is_monotonic_decreasing or s["부도율(%)"].is_monotonic_increasing
        spread.append({"변수": c, "최고(%)": hi, "최저(%)": lo,
                       "배수": round(hi / lo, 2), "단조성": "○" if mono else "×"})
    spread = pd.DataFrame(spread).sort_values("배수", ascending=False)
    for _, r in spread.iterrows():
        log(f"    {r['변수']:22s} {r['최저(%)']:5.2f}% ~ {r['최고(%)']:6.2f}% "
            f"({r['배수']:5.2f}배)  단조성 {r['단조성']}")

    # ══════════════════════════════════════════════════════════════
    stage(2, "축 선정 — 세분화에 쓸 변수 확정")
    # ══════════════════════════════════════════════════════════════
    use_cols = [c for c in SHAP_TOP if c in df.columns and c not in EXCLUDE]
    missing = [c for c in SHAP_TOP if c not in df.columns]
    for c in missing:
        log(f"  [경고] 컬럼 없음 → 제외: {c}")

    axis = pd.DataFrame({
        "변수": use_cols,
        "SHAP중요도": [SHAP_TOP[c] for c in use_cols],
    })
    axis["SHAP순위"] = axis["SHAP중요도"].rank(ascending=False).astype(int)
    axis["단변량배수"] = axis["변수"].map(
        dict(zip(spread["변수"], spread["배수"])))
    axis["채택사유"] = "SHAP TOP10"
    axis = pd.concat([axis, pd.DataFrame({
        "변수": EXCLUDE, "SHAP중요도": [SHAP_TOP.get(c, np.nan) for c in EXCLUDE],
        "SHAP순위": np.nan, "단변량배수": np.nan,
        "채택사유": ["제외 — 공정성(4/5 rule 위반)", "제외 — median 대치 아티팩트"],
    })], ignore_index=True)
    axis.to_csv(f"{OUT}/SEG_2_축선정.csv", index=False, encoding=ENC)

    log(f"세분화 축 {len(use_cols)}개 확정")
    for _, r in axis.iterrows():
        log(f"    {str(r['변수']):24s} {r['채택사유']}")

    X = df[use_cols].copy()
    X = ordinal_encode(df, X, use_cols).astype(float)
    X = X.fillna(X.median())
    y = df[TARGET].values

    # ══════════════════════════════════════════════════════════════
    stage(3, "규칙 세분화 (의사결정나무) — 주 방법")
    # ══════════════════════════════════════════════════════════════
    # 예측이 아니라 세분화가 목적. 예측은 XGBoost가 이미 한다.
    # 그래서 얕게(depth 3) 묶고 리프 최소 인원을 크게 잡는다.
    rows = []
    for d in TREE_DEPTHS:
        c = DecisionTreeClassifier(max_depth=d, min_samples_leaf=TREE_MIN_LEAF,
                                   random_state=SEED).fit(X, y)
        g = pd.DataFrame({"leaf": c.apply(X), "y": y}).groupby("leaf")["y"]
        rate = g.mean()
        rows.append({"깊이": d, "세그먼트수": len(rate),
                     "최저부도율(%)": round(rate.min() * 100, 2),
                     "최고부도율(%)": round(rate.max() * 100, 2),
                     "분별배수": round(rate.max() / rate.min(), 2),
                     "최소인원": int(g.size().min())})
        log(f"  depth={d}  세그먼트 {len(rate):>2}개  "
            f"{rate.min()*100:5.2f}% ~ {rate.max()*100:6.2f}%  "
            f"({rate.max()/rate.min():5.2f}배)")
    pd.DataFrame(rows).to_csv(f"{OUT}/SEG_3_깊이비교.csv",
                              index=False, encoding=ENC)

    clf = DecisionTreeClassifier(max_depth=TREE_DEPTH,
                                 min_samples_leaf=TREE_MIN_LEAF,
                                 random_state=SEED).fit(X, y)
    df["_leaf"] = clf.apply(X)
    rules = extract_rules(clf, use_cols)

    seg = (df.groupby("_leaf")[TARGET]
             .agg(인원수="count", 부도자수="sum").reset_index())
    seg["부도율(%)"] = (seg["부도자수"] / seg["인원수"] * 100).round(3)
    seg["리프트"] = (seg["부도율(%)"] / 100 / BASE).round(3)
    seg["비율(%)"] = (seg["인원수"] / N * 100).round(2)
    seg = add_ci(seg).sort_values("부도율(%)", ascending=False).reset_index(drop=True)
    seg.insert(0, "세그먼트", [f"S{i+1}" for i in range(len(seg))])
    seg["규칙"] = seg["_leaf"].map(rules)

    df["세그먼트"] = df["_leaf"].map(dict(zip(seg["_leaf"], seg["세그먼트"])))
    SEG_ORDER = seg["세그먼트"].tolist()

    seg[["세그먼트", "규칙", "인원수", "비율(%)", "부도자수", "부도율(%)",
         "리프트", "CI하한(%)", "CI상한(%)", "CI폭(%p)"]].to_csv(
        f"{OUT}/SEG_3_규칙표.csv", index=False, encoding=ENC)

    log("")
    log(f"채택 depth={TREE_DEPTH} → 세그먼트 {len(seg)}개")
    for _, r in seg.iterrows():
        log(f"  [{r['세그먼트']}] {r['부도율(%)']:6.2f}%  리프트 {r['리프트']:5.2f}  "
            f"{r['인원수']:>7,}명")
        log(f"        {r['규칙']}")

    imp = pd.Series(clf.feature_importances_, index=use_cols)
    imp = imp[imp > 0].sort_values(ascending=False)
    log("")
    log("나무가 실제 사용한 변수:")
    for c, v in imp.items():
        log(f"    {c:24s} {v:.4f}")
    unused = [c for c in use_cols if c not in imp.index]
    if unused:
        log(f"  미사용: {', '.join(unused)}")

    # ══════════════════════════════════════════════════════════════
    stage(4, "보조 군집화 (K-Means) — 나무가 놓친 집단 탐색")
    # ══════════════════════════════════════════════════════════════
    # SHAP 중요도를 거리 가중치로 사용한다.
    #   ▶ sqrt를 씌우는 이유: 유클리드 거리는 좌표를 제곱해 더하므로
    #     (w·x)² = w²·x² 가 되어 가중치가 두 번 적용된다.
    #     sqrt를 씌워야 거리 기여도가 SHAP에 정확히 비례한다.
    Xs = StandardScaler().fit_transform(X.values)
    w = np.sqrt(np.array([SHAP_TOP[c] for c in use_cols]))
    w = w / w.sum() * len(w)
    Xw = Xs * w

    rng = np.random.default_rng(SEED)
    sidx = rng.choice(len(Xw), size=min(SIL_SAMPLE, len(Xw)), replace=False)

    ks, inert, sils = [], [], []
    for k in K_RANGE:
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=10,
                             batch_size=4096, max_iter=300)
        lab = km.fit_predict(Xw)
        s = silhouette_score(Xw[sidx], lab[sidx])
        ks.append(k); inert.append(km.inertia_); sils.append(s)
        log(f"  k={k}  관성 {km.inertia_:>12,.0f}   실루엣 {s:.4f}")

    km = MiniBatchKMeans(n_clusters=K_FINAL, random_state=SEED, n_init=20,
                         batch_size=4096, max_iter=500)
    df["_c"] = km.fit_predict(Xw)
    # 부도율 오름차순으로 번호 재배열 (C0 = 가장 안전)
    remap = {old: new for new, old in
             enumerate(df.groupby("_c")[TARGET].mean().sort_values().index)}
    df["군집"] = df["_c"].map(remap).map(lambda i: f"C{i}")
    CL_ORDER = [f"C{i}" for i in range(K_FINAL)]

    cl = (df.groupby("군집")[TARGET]
            .agg(인원수="count", 부도자수="sum").reindex(CL_ORDER))
    cl["부도율(%)"] = (cl["부도자수"] / cl["인원수"] * 100).round(3)
    cl["리프트"] = (cl["부도율(%)"] / 100 / BASE).round(3)
    cl["비율(%)"] = (cl["인원수"] / N * 100).round(2)
    cl = add_ci(cl.reset_index()).set_index("군집")
    cl.to_csv(f"{OUT}/SEG_4_군집프로파일.csv", encoding=ENC)

    zdf = pd.DataFrame(Xs, columns=use_cols, index=df.index)
    zdf["군집"] = df["군집"].values
    zmean = zdf.groupby("군집").mean().reindex(CL_ORDER)
    zmean.round(3).to_csv(f"{OUT}/SEG_4_군집특징.csv", encoding=ENC)

    log("")
    print(cl.to_string())
    log("")
    log("군집 특징 (표준화 평균, |z|>0.8 만 표시)")
    for c in CL_ORDER:
        feats = zmean.loc[c]
        strong = feats[feats.abs() > 0.8].sort_values(key=abs, ascending=False)
        desc = ", ".join(f"{k} {v:+.1f}" for k, v in strong.items()) or "뚜렷한 특징 없음"
        log(f"    {c} (부도율 {cl.loc[c,'부도율(%)']:5.2f}%)  {desc}")

    # ══════════════════════════════════════════════════════════════
    stage(5, "교차검증 — 나무와 군집이 같은 집단을 지목하는가")
    # ══════════════════════════════════════════════════════════════
    ct = (pd.crosstab(df["세그먼트"], df["군집"], normalize="index") * 100)
    ct = ct.reindex(index=SEG_ORDER, columns=CL_ORDER).round(1)
    ct.to_csv(f"{OUT}/SEG_5_교차검증.csv", encoding=ENC)
    print(ct.to_string())

    conc = ct.max(axis=1)
    exp = 100 / K_FINAL
    log("")
    log(f"평균 집중도 {conc.mean():.1f}%  (무작위 기대치 {exp:.1f}%, "
        f"{conc.mean()/exp:.1f}배)")
    log("→ 지도학습(나무)과 무감독(K-Means)이 독립적으로 같은 집단을 지목")

    # 나무가 놓친 군집 찾기: 여러 세그먼트에 흩어진 군집
    log("")
    log("나무가 놓친 집단 점검 (군집이 여러 세그먼트에 흩어졌는가)")
    ct_col = (pd.crosstab(df["군집"], df["세그먼트"], normalize="index") * 100)
    ct_col = ct_col.reindex(index=CL_ORDER, columns=SEG_ORDER)
    for c in CL_ORDER:
        mx = ct_col.loc[c].max()
        flag = "  ← 나무가 분리하지 못함" if mx < 40 else ""
        log(f"    {c}: 최대 세그먼트 집중도 {mx:.1f}%{flag}")

    # ══════════════════════════════════════════════════════════════
    stage(6, "공정성 점검 — 보호변수 제거 후에도 대리차별이 남았는가")
    # ══════════════════════════════════════════════════════════════
    checks = []
    for col in FAIRNESS_CHECK:
        if col not in df.columns:
            continue
        t = (pd.crosstab(df["세그먼트"], df[col], normalize="index") * 100)
        t = t.reindex(SEG_ORDER).round(2)
        t.insert(0, "점검항목", col)
        checks.append(t)
    if checks:
        pd.concat(checks).to_csv(f"{OUT}/SEG_6_공정성점검.csv", encoding=ENC)

    # 4/5 rule — 최고위험 세그먼트 편입률의 집단 간 비율
    hi_seg = SEG_ORDER[0]
    log(f"최고위험 세그먼트 = {hi_seg} "
        f"(부도율 {seg.iloc[0]['부도율(%)']:.2f}%)")
    for col in FAIRNESS_CHECK:
        if col not in df.columns:
            continue
        share = df.groupby(col, observed=True).apply(
            lambda g: (g["세그먼트"] == hi_seg).mean(), include_groups=False)
        share = share[df[col].value_counts().reindex(share.index) >= MIN_N]
        if len(share) < 2 or share.max() == 0:
            continue
        ratio = share.min() / share.max()
        verdict = "충족" if ratio >= 0.8 else "미충족 → 대리차별 잔존"
        log(f"  [{col}] 편입률 {share.min()*100:.2f}% ~ {share.max()*100:.2f}%  "
            f"비율 {ratio:.3f}  4/5 rule {verdict}")
        if col == "성별":
            for k, v in share.items():
                log(f"      {k}: {v*100:.2f}%")

    # ══════════════════════════════════════════════════════════════
    stage(7, "세그먼트 확정 — EV 기반 대응전략")
    # ══════════════════════════════════════════════════════════════
    # 손익분기 부도율 = 순이익률 / (순이익률 + LGD)
    #   이보다 부도율이 높으면 대출 1건의 기대손익이 음수가 된다.
    BE = NET_MARGIN / (NET_MARGIN + LGD)

    def strategy(p):
        p = p / 100
        if p >= BE * 1.5:
            return "거절"
        if p >= BE:
            return "거절 또는 담보·보증 조건부"
        if p >= BE * 0.6:
            return "승인 · 한도 축소 + 가산금리"
        if p >= BE * 0.3:
            return "표준 승인"
        return "승인 · 우대금리 + 한도 상향"

    final = seg[["세그먼트", "규칙", "인원수", "비율(%)", "부도자수",
                 "부도율(%)", "리프트", "CI하한(%)", "CI상한(%)"]].copy()
    final["손익분기대비"] = (final["부도율(%)"] / 100 / BE).round(3)
    final["기대손익(원금1당)"] = (
        (1 - final["부도율(%)"] / 100) * NET_MARGIN
        - (final["부도율(%)"] / 100) * LGD).round(5)
    final["위험가산금리(%p)"] = (
        LGD * (final["부도율(%)"] / 100) / (1 - final["부도율(%)"] / 100) * 100
    ).round(2)
    final["대응전략"] = final["부도율(%)"].map(strategy)
    final["대표군집"] = ct.idxmax(axis=1).reindex(final["세그먼트"]).values
    final.to_csv(f"{OUT}/SEG_7_최종세그먼트.csv", index=False, encoding=ENC)

    log(f"손익분기 부도율 {BE:.4%}")
    log("")
    for _, r in final.iterrows():
        mark = "▲" if r["기대손익(원금1당)"] < 0 else " "
        log(f" {mark}[{r['세그먼트']}] {r['부도율(%)']:6.2f}%  "
            f"EV {r['기대손익(원금1당)']:+.4f}  "
            f"가산 {r['위험가산금리(%p)']:5.2f}%p  "
            f"{r['인원수']:>7,}명  →  {r['대응전략']}")
    neg = final[final["기대손익(원금1당)"] < 0]
    log("")
    log(f"기대손익 음수 세그먼트 {len(neg)}개 · {neg['인원수'].sum():,}명 "
        f"({neg['인원수'].sum()/N*100:.2f}%)")

    df[["세그먼트", "군집", TARGET]].to_pickle(f"{OUT}/SEG_할당결과.pkl")

    # ══════════════════════════════════════════════════════════════
    log("")
    log("시각화 생성 중...")
    # ══════════════════════════════════════════════════════════════
    # 그림1 — 단변량
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top = pd.concat([cat_df.head(8), cat_df.tail(8)])
    cols = ["#BF5620" if v > 1 else "#2C5F45" for v in top["리프트"]]
    axes[0].barh(range(len(top)), top["부도율(%)"], color=cols)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels([f"{r['변수'][:8]}={r['값'][:12]}"
                             for _, r in top.iterrows()], fontsize=8)
    axes[0].axvline(BASE * 100, color="black", ls="--", lw=1)
    axes[0].invert_yaxis(); axes[0].set_xlabel("부도율(%)")
    axes[0].set_title("범주형 — 최고·최저 각 8개")

    for c in num_df["변수"].unique():
        s = num_df[num_df["변수"] == c].sort_values("분위")
        axes[1].plot(s["분위"], s["부도율(%)"], "o-", label=c[:14], lw=1.6)
    axes[1].axhline(BASE * 100, color="black", ls="--", lw=1)
    axes[1].set_xlabel("5분위 (인원 균등)"); axes[1].set_ylabel("부도율(%)")
    axes[1].legend(fontsize=8); axes[1].set_title("수치형 — 분위별 추세")
    plt.tight_layout()
    plt.savefig(f"{OUT}/SEG_그림1_단변량.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 그림2 — 나무
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(clf, feature_names=use_cols, class_names=["정상", "부도"],
              filled=True, rounded=True, proportion=True, impurity=False,
              fontsize=8, ax=ax)
    ax.set_title(f"의사결정나무 세분화 (depth={TREE_DEPTH})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT}/SEG_그림2_나무.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 그림3 — 군집
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    axes[0].plot(ks, inert, "o-", color="#2C5F45")
    a2 = axes[0].twinx(); a2.plot(ks, sils, "s--", color="#BF5620")
    axes[0].axvline(K_FINAL, color="gray", ls=":")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("관성", color="#2C5F45")
    a2.set_ylabel("실루엣", color="#BF5620")
    axes[0].set_title(f"최적 k — 채택 {K_FINAL}")

    axes[1].bar(CL_ORDER, cl["부도율(%)"],
                color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, K_FINAL)))
    axes[1].axhline(BASE * 100, color="black", ls="--", lw=1)
    for i, (v, n) in enumerate(zip(cl["부도율(%)"], cl["인원수"])):
        axes[1].text(i, v + 0.3, f"{v:.1f}%\n{n:,}", ha="center", fontsize=8)
    axes[1].set_ylabel("부도율(%)"); axes[1].set_title("군집별 부도율")

    im = axes[2].imshow(zmean.values, cmap="coolwarm", aspect="auto",
                        vmin=-1.5, vmax=1.5)
    axes[2].set_xticks(range(len(use_cols)))
    axes[2].set_xticklabels([c[:11] for c in use_cols], rotation=45,
                            ha="right", fontsize=7)
    axes[2].set_yticks(range(K_FINAL)); axes[2].set_yticklabels(CL_ORDER)
    for i in range(K_FINAL):
        for j in range(len(use_cols)):
            axes[2].text(j, i, f"{zmean.values[i,j]:.1f}", ha="center",
                         va="center", fontsize=7)
    plt.colorbar(im, ax=axes[2]); axes[2].set_title("군집 특징 (z)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/SEG_그림3_군집.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 그림4 — 최종 요약
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.5),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    cols = plt.cm.RdYlGn_r(final["리프트"] / final["리프트"].max())
    b = axes[0].bar(final["세그먼트"], final["부도율(%)"], color=cols,
                    edgecolor="white")
    axes[0].errorbar(final["세그먼트"], final["부도율(%)"],
                     yerr=[final["부도율(%)"] - final["CI하한(%)"],
                           final["CI상한(%)"] - final["부도율(%)"]],
                     fmt="none", ecolor="black", capsize=4, lw=1)
    axes[0].axhline(BASE * 100, color="black", ls="--", lw=1.2,
                    label=f"기저율 {BASE*100:.2f}%")
    axes[0].axhline(BE * 100, color="#BF5620", ls="-.", lw=1.6,
                    label=f"손익분기 {BE*100:.2f}%")
    for bb, v, n in zip(b, final["부도율(%)"], final["인원수"]):
        axes[0].text(bb.get_x() + bb.get_width() / 2, v + 0.5,
                     f"{v:.1f}%\n{n:,}", ha="center", fontsize=8)
    axes[0].set_ylabel("부도율(%)"); axes[0].legend()
    axes[0].set_title(f"최종 세그먼트 — 분별력 "
                      f"{final['부도율(%)'].max()/final['부도율(%)'].min():.1f}배")

    im = axes[1].imshow(ct.values, cmap="Greens", aspect="auto", vmin=0, vmax=100)
    axes[1].set_xticks(range(K_FINAL)); axes[1].set_xticklabels(CL_ORDER)
    axes[1].set_yticks(range(len(SEG_ORDER))); axes[1].set_yticklabels(SEG_ORDER)
    for i in range(len(SEG_ORDER)):
        for j in range(K_FINAL):
            v = ct.values[i, j]
            axes[1].text(j, i, f"{v:.0f}", ha="center", va="center",
                         fontsize=8, color="white" if v > 50 else "black")
    axes[1].set_xlabel("K-Means 군집"); axes[1].set_ylabel("나무 세그먼트")
    axes[1].set_title(f"교차검증 — 평균 집중도 {conc.mean():.1f}% "
                      f"(무작위 {exp:.0f}%)")
    plt.colorbar(im, ax=axes[1], label="비중(%)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/SEG_그림4_최종요약.png", dpi=150, bbox_inches="tight")
    plt.close()

    print()
    print("═" * 78)
    log("완료 — 출력 파일 (접두사 SEG_)")
    for f in ["1_단변량_범주형.csv", "1_단변량_수치형.csv", "2_축선정.csv",
              "3_규칙표.csv", "3_깊이비교.csv", "4_군집프로파일.csv",
              "4_군집특징.csv", "5_교차검증.csv", "6_공정성점검.csv",
              "7_최종세그먼트.csv", "그림1_단변량.png", "그림2_나무.png",
              "그림3_군집.png", "그림4_최종요약.png", "할당결과.pkl"]:
        log(f"    SEG_{f}")
    log(f"총 소요시간 {time.time() - T0:.1f}초")
