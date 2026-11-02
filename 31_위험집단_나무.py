# -*- coding: utf-8 -*-
"""
STEP 31. SHAP 상위 변수로 위험 집단 찾기 (의사결정나무)  — v2

원하는 결과
    "신용점수 매우낮음 · 계약금비율 낮음 · 연체이력 잦음
       →  부도율 26.0%  (평균의 3.2배, 10,479명)"

■ 왜 변수를 등급으로 바꾸는가
  SHAP TOP10은 대부분 숫자 변수다. 그대로 넣으면 조건이
  "외부신용점수_평균 <= 0.4164"처럼 나와 사람이 못 읽는다.
  각 변수를 3~5단계 등급으로 먼저 바꾸면 규칙이 그대로 문장이 된다.

════════════════════════════════════════════════════════════════
v1에서 고친 3가지
════════════════════════════════════════════════════════════════
① 부정 조건을 긍정으로 변환
   v1:  "신용점수 매우낮음/낮음/보통/매우높음 아님"   ← 5개 중 4개를 부정
   v2:  "신용점수 높음"                              ← 남은 하나를 긍정으로
   3단계 변수의 "낮음 아님"도 "보통~높음"으로 바꾼다.
   남은 값이 연속 구간이면 'A~B', 흩어져 있으면 'A/B'로 표기.

② 연간상환액 제외
   등급별 부도율이 적음 7.61% / 보통 8.94% / 많음 7.67%로 역U자다.
   "많으면 위험"도 "적으면 위험"도 말할 수 없어 규칙에 들어가면
   해석이 불가능하다. Q&A에서 방어가 안 되므로 축에서 뺀다.

③ 깊이 4 → 3
   v1은 집단이 16개 나와 슬라이드 한 장에 안 들어간다.
   깊이 3이면 8개 안팎으로 줄어든다.

   추가로 계약금비율은 1.0에서 자른다.
   비율이 1.0을 넘는 506명은 계약금이 대출금보다 크다는 뜻이라
   데이터 오류 가능성이 높은데, 표본이 작아 부도율이 불안정하다.

■ 성별 제외
  SHAP 3위(0.117)지만 20번 공정성 검증에서 4/5 rule 위반(0.557).
  분할 축으로 쓰면 "여성이면 위험" 같은 규칙이 그대로 출력된다.

출력
    위험집단_등급기준표.csv   각 변수의 등급 경계값
    위험집단_규칙.csv         집단별 규칙 · 부도율 · 인원 · 신뢰구간
    위험집단_요약.txt         문장으로 정리  ← 이것만 봐도 됨
    위험집단_카드.csv         TOP3 / BOTTOM3 발표용
    위험집단_카드.png         TOP3 / BOTTOM3 카드 이미지  ← PPT용
    위험집단_나무.png         나무 구조도
    위험집단_그래프.png       집단별 부도율 막대
"""
import time
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import DecisionTreeClassifier, plot_tree

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
SEED = 42

DEPTH = 3          # v1의 4에서 낮춤 (집단 16개 → 8개 안팎)
MIN_LEAF = 1500
TOP_N = 3          # 카드로 뽑을 상·하위 개수
USE_GENDER = False

# ── SHAP 상위 변수 → 등급  (원본, 표시명, 등급수, 라벨, 상한클립) ──
#    연간상환액은 역U자라 v2에서 제외했다(상단 주석 ② 참조)
BANDS = [
    ("외부신용점수_평균",       "신용점수",   5,
     ["매우낮음", "낮음", "보통", "높음", "매우높음"], None),
    ("신용부담비율_상환대출",    "상환부담",   3, ["낮음", "보통", "높음"], None),
    ("외부신용점수_최솟값",      "신용최저점", 3, ["낮음", "보통", "높음"], None),
    ("계약금비율",             "계약금비율", 3, ["낮음", "보통", "높음"], 1.0),
    ("INST_실제상환금액_SUM",   "누적상환액", 3, ["적음", "보통", "많음"], None),
    ("POS_잔여할부개월수_MEAN", "잔여할부",   3, ["짧음", "보통", "김"], None),
]
LATE_COL = "INST_IS_LATE_MEAN"
EDU_COL = "최종학력"

# 학력은 5단계 → 3단계로 합친다.
#   v2에서 나무가 '중등교육이냐 아니냐'만 보는 바람에
#   중학교이하(부도율 10.93%)와 석박사(1.83%)가 한 집단으로 묶여
#   "학력 중학교이하/대학중퇴/고등교육/석박사학위"라는 발표 불가능한
#   규칙이 나왔다. 부도율이 비슷한 것끼리 미리 묶어 이를 막는다.
EDU_MERGE = {
    "중학교이하": "고졸이하",
    "중등교육": "고졸이하",
    "대학중퇴": "대학중퇴",
    "고등교육(대졸이상)": "대졸이상",
    "석박사학위": "대졸이상",
}
EDU_ORDER = ["고졸이하", "대학중퇴", "대졸이상"]

SAFE_MAX_LIFT = 0.8   # 안전 카드는 평균의 0.8배 이하만 (평균 수준은 '안전'이 아님)

# 값이 순서를 가지면 'A~B'로, 아니면 'A/B'로 묶는다
ORDERED = set()
# ════════════════════════════════════════════════════════════════


def log(m=""):
    print(f"[{time.strftime('%H:%M:%S')}] {m}" if m else "")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def make_band(s, n, labels):
    """인원 균등분할. 중복 경계로 구간이 줄면 라벨을 균등 배분."""
    b = pd.qcut(s, n, duplicates="drop")
    cats = b.cat.categories
    if len(cats) == len(labels):
        return b.cat.rename_categories(labels), cats
    idx = np.linspace(0, len(labels) - 1, len(cats)).round().astype(int)
    return b.cat.rename_categories([labels[i] for i in idx]), cats


def rules_to_korean(clf, feat_names, levels):
    """리프까지의 경로를 한국어 문장으로.

    원-핫이므로 각 분기는 '변수=값'의 참·거짓이다.
      > 0.5  →  그 값으로 확정   (pos)
     <= 0.5  →  그 값이 아님     (neg)

    ▶ v2 핵심: neg만 쌓인 변수는 '남은 값'을 계산해 긍정으로 표현한다.
       levels = {변수: [가능한 값들(순서대로)]}
       남은 값이 1개면 그 값, 연속 구간이면 'A~B', 흩어지면 'A/B'.
    """
    t = clf.tree_
    out = {}

    def describe(var, remain):
        lv = levels[var]
        idx = sorted(lv.index(v) for v in remain)
        if len(idx) == 1:
            return f"{var} {lv[idx[0]]}"
        if len(idx) == len(lv):
            return None                      # 제약 없음 → 표시 불필요
        contiguous = idx == list(range(idx[0], idx[-1] + 1))
        if contiguous and var in ORDERED:
            return f"{var} {lv[idx[0]]}~{lv[idx[-1]]}"
        return f"{var} {'/'.join(lv[i] for i in idx)}"

    def fmt(pos, neg):
        parts = []
        for var, val in pos.items():
            parts.append(f"{var} {val}")
        for var, excluded in neg.items():
            if var in pos:                   # 이미 확정 → 부정 조건은 군더더기
                continue
            remain = [v for v in levels[var] if v not in excluded]
            if not remain:
                continue
            d = describe(var, remain)
            if d:
                parts.append(d)
        return "  ·  ".join(parts) if parts else "(전체)"

    def walk(node, pos, neg):
        if t.children_left[node] == -1:
            out[node] = fmt(pos, neg)
            return
        var, val = feat_names[t.feature[node]].split("=", 1)

        n2 = {k: list(v) for k, v in neg.items()}
        n2.setdefault(var, []).append(val)
        walk(t.children_left[node], dict(pos), n2)

        p2 = dict(pos)
        p2[var] = val
        walk(t.children_right[node], p2, {k: list(v) for k, v in neg.items()})

    walk(0, {}, {})
    return out


if __name__ == "__main__":
    t0 = time.time()

    df = pd.read_pickle(f"{OUT}/full_feature_table_derived.pkl")
    df = df[df["IS_TRAIN"] == 1].copy()
    df[TARGET] = df[TARGET].astype(int)
    BASE = df[TARGET].mean()
    log(f"{len(df):,}명 · 평균 부도율 {BASE*100:.2f}%")

    # ══════════════════════════════════════════════════════════
    log("")
    log("1) SHAP 상위 변수 → 등급 변환")
    # ══════════════════════════════════════════════════════════
    band_cols, ref, LEVELS = [], [], {}

    for src, name, n, labels, clip in BANDS:
        if src not in df.columns:
            log(f"  [경고] 컬럼 없음 → 제외: {src}")
            continue
        s = df[src]
        if clip is not None:
            over = int((s > clip).sum())
            if over:
                s = s.clip(upper=clip)
                log(f"  {name}: {clip} 초과 {over:,}명을 {clip}로 클립 "
                    f"(데이터 오류 가능성)")
        df[name], cats = make_band(s, n, labels)
        band_cols.append(name)
        LEVELS[name] = df[name].cat.categories.tolist()
        ORDERED.add(name)
        for lab, iv in zip(LEVELS[name], cats):
            m = df[name] == lab
            ref.append({"표시이름": name, "원본컬럼": src, "등급": lab,
                        "구간": str(iv), "인원수": int(m.sum()),
                        "부도율(%)": round(df.loc[m, TARGET].mean() * 100, 2)})
        log(f"  {name:10s} ← {src}  ({' / '.join(LEVELS[name])})")

    # 연체이력 — 0인 사람이 많아 균등분할 불가, 별도 처리
    if LATE_COL in df.columns:
        s = df[LATE_COL].fillna(0)
        pos = s[s > 0]
        cut = pos.median() if len(pos) else 0
        df["연체이력"] = pd.Categorical(
            np.where(s <= 0, "없음", np.where(s <= cut, "가끔", "잦음")),
            categories=["없음", "가끔", "잦음"], ordered=True)
        band_cols.append("연체이력")
        LEVELS["연체이력"] = ["없음", "가끔", "잦음"]
        ORDERED.add("연체이력")
        for lab, desc in [("없음", "= 0"), ("가끔", f"0 초과 ~ {cut:.4g}"),
                          ("잦음", f"{cut:.4g} 초과")]:
            m = df["연체이력"] == lab
            ref.append({"표시이름": "연체이력", "원본컬럼": LATE_COL, "등급": lab,
                        "구간": desc, "인원수": int(m.sum()),
                        "부도율(%)": round(df.loc[m, TARGET].mean() * 100, 2)})
        log(f"  연체이력     ← {LATE_COL}  (없음 / 가끔 / 잦음)")

    # 최종학력 — 5단계를 3단계로 합친 뒤 교육 수준 순으로 정렬
    if EDU_COL in df.columns:
        raw = df[EDU_COL].astype(str)
        unmapped = sorted(set(raw) - set(EDU_MERGE))
        if unmapped:
            log(f"  [주의] 학력 매핑에 없는 값 → 그대로 사용: {unmapped}")
        vals = raw.map(lambda v: EDU_MERGE.get(v, v))
        order = [v for v in EDU_ORDER if v in set(vals)]
        order += [v for v in vals.unique() if v not in order]
        df["학력"] = pd.Categorical(vals, categories=order, ordered=True)
        band_cols.append("학력")
        LEVELS["학력"] = order
        ORDERED.add("학력")
        for lab in order:
            m = df["학력"] == lab
            ref.append({"표시이름": "학력", "원본컬럼": EDU_COL, "등급": lab,
                        "구간": "(범주형)", "인원수": int(m.sum()),
                        "부도율(%)": round(df.loc[m, TARGET].mean() * 100, 2)})
        log(f"  학력         ← {EDU_COL}  ({' < '.join(order)})")

    if USE_GENDER and "성별" in df.columns:
        df["성별_"] = df["성별"].astype(str)
        band_cols.append("성별_")
        LEVELS["성별_"] = sorted(df["성별_"].unique())
        log("  성별  ← 포함됨 (공정성 위반 변수 — PPT 사용 금지)")

    log(f"  ※ 연간상환액은 등급별 부도율이 역U자(7.61/8.94/7.67%)라 제외")

    ref_df = pd.DataFrame(ref)
    ref_df.to_csv(f"{OUT}/위험집단_등급기준표.csv", index=False, encoding=ENC)

    print()
    print("─" * 78)
    print("  [등급 기준표] — '낮음/높음'이 실제로 어떤 값인지")
    print("─" * 78)
    for name in ref_df["표시이름"].unique():
        print(f"  {name}")
        for _, r in ref_df[ref_df["표시이름"] == name].iterrows():
            print(f"      {r['등급']:12s} {r['구간']:32s} "
                  f"{r['인원수']:>7,}명  부도율 {r['부도율(%)']:5.2f}%")

    # ══════════════════════════════════════════════════════════
    log("")
    log(f"2) 나무 학습 (깊이 {DEPTH} · 집단 최소 {MIN_LEAF:,}명)")
    # ══════════════════════════════════════════════════════════
    X = pd.get_dummies(df[band_cols].astype(str), prefix_sep="=").astype(float)
    y = df[TARGET].values
    log(f"   {len(band_cols)}개 등급변수 → 원-핫 {X.shape[1]}개 컬럼")

    clf = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF,
                                 random_state=SEED).fit(X, y)
    df["_leaf"] = clf.apply(X)
    rules = rules_to_korean(clf, list(X.columns), LEVELS)

    g = df.groupby("_leaf")[TARGET].agg(["count", "sum"]).reset_index()
    res = pd.DataFrame({"_leaf": g["_leaf"],
                        "인원수": g["count"].astype(int),
                        "부도자수": g["sum"].astype(int)})
    res["부도율(%)"] = (res["부도자수"] / res["인원수"] * 100).round(2)
    res["평균대비배수"] = (res["부도율(%)"] / 100 / BASE).round(2)
    res["비율(%)"] = (res["인원수"] / len(df) * 100).round(2)
    ci = [wilson(int(a), int(b)) for a, b in zip(res["부도자수"], res["인원수"])]
    res["CI하한(%)"] = [round(a * 100, 2) for a, _ in ci]
    res["CI상한(%)"] = [round(b * 100, 2) for _, b in ci]
    res["집단설명"] = res["_leaf"].map(rules)
    res = res.sort_values("부도율(%)", ascending=False).reset_index(drop=True)
    res.insert(0, "순위", range(1, len(res) + 1))

    res[["순위", "집단설명", "인원수", "비율(%)", "부도자수", "부도율(%)",
         "평균대비배수", "CI하한(%)", "CI상한(%)"]].to_csv(
        f"{OUT}/위험집단_규칙.csv", index=False, encoding=ENC)

    imp = pd.Series(clf.feature_importances_, index=X.columns)
    imp = imp[imp > 0].sort_values(ascending=False)
    by_var = imp.groupby(imp.index.str.split("=").str[0]).sum().sort_values(
        ascending=False)
    used = set(by_var.index)
    unused = [c for c in band_cols if c not in used]

    # ══════════════════════════════════════════════════════════
    # 3) TOP / BOTTOM 카드
    # ══════════════════════════════════════════════════════════
    top = res.head(TOP_N).copy()
    # 안전 카드는 '평균의 SAFE_MAX_LIFT배 이하'만 인정한다.
    #   v2에서는 하위 3개를 무조건 뽑는 바람에 평균과 같은 8.08%(리프트 1.00)가
    #   "안전 3위"로 표시됐다. 평균 수준은 안전이 아니다.
    safe = res[res["평균대비배수"] <= SAFE_MAX_LIFT]
    bot = safe.tail(TOP_N).iloc[::-1].copy()
    log(f"   안전 후보 {len(safe)}개 (리프트 ≤ {SAFE_MAX_LIFT}) → 카드 {len(bot)}장")
    top["구분"] = "위험"
    bot["구분"] = "안전"
    cards = pd.concat([top, bot], ignore_index=True)
    cards[["구분", "순위", "집단설명", "인원수", "비율(%)", "부도율(%)",
           "평균대비배수", "CI하한(%)", "CI상한(%)"]].to_csv(
        f"{OUT}/위험집단_카드.csv", index=False, encoding=ENC)

    # ══════════════════════════════════════════════════════════
    # 4) 문장 정리
    # ══════════════════════════════════════════════════════════
    L = [f"전체 {len(df):,}명 · 평균 부도율 {BASE*100:.2f}%",
         f"집단 {len(res)}개 · 분별력 "
         f"{res['부도율(%)'].max()/res['부도율(%)'].min():.1f}배 "
         f"({res['부도율(%)'].min():.2f}% ~ {res['부도율(%)'].max():.2f}%)",
         "▲ = 평균의 1.3배 이상 위험   ▽ = 0.7배 이하 안전"]

    L += ["", "=" * 78, f"  ★ 가장 위험한 {TOP_N}개 집단", "=" * 78]
    for i, (_, r) in enumerate(top.iterrows(), 1):
        L.append(f"  {i}. {r['집단설명']}")
        L.append(f"       부도율 {r['부도율(%)']:5.2f}%  "
                 f"(평균의 {r['평균대비배수']:.2f}배)  "
                 f"{r['인원수']:,}명 · 전체의 {r['비율(%)']:.1f}%")
        L.append(f"       95% 신뢰구간 {r['CI하한(%)']:.2f}% ~ {r['CI상한(%)']:.2f}%")

    L += ["", "=" * 78,
          f"  ★ 가장 안전한 {len(bot)}개 집단  (평균의 {SAFE_MAX_LIFT}배 이하만 인정)",
          "=" * 78]
    for i, (_, r) in enumerate(bot.iterrows(), 1):
        L.append(f"  {i}. {r['집단설명']}")
        L.append(f"       부도율 {r['부도율(%)']:5.2f}%  "
                 f"(평균의 {r['평균대비배수']:.2f}배)  "
                 f"{r['인원수']:,}명 · 전체의 {r['비율(%)']:.1f}%")
        L.append(f"       95% 신뢰구간 {r['CI하한(%)']:.2f}% ~ {r['CI상한(%)']:.2f}%")

    L += ["", "=" * 78, "  전체 집단 (위험한 순서)", "=" * 78]
    for _, r in res.iterrows():
        mark = "▲" if r["평균대비배수"] >= 1.3 else (
               "▽" if r["평균대비배수"] <= 0.7 else " ")
        L.append(f"{mark} {r['순위']:2}. {r['집단설명']}")
        L.append(f"       → 부도율 {r['부도율(%)']:5.2f}%  "
                 f"(평균의 {r['평균대비배수']:.2f}배)  "
                 f"{r['인원수']:,}명 ({r['비율(%)']:.1f}%)")

    L += ["", "=" * 78, "  변수별 기여도 (원-핫 컬럼을 변수 단위로 합산)", "=" * 78]
    for c, v in by_var.items():
        L.append(f"    {c:12s} {v:.4f}  {'█' * int(v * 50)}")
    if unused:
        L += ["", f"  한 번도 안 쓰인 변수: {', '.join(unused)}",
              "  → SHAP에서는 중요했지만 집단을 나누는 수준에서는",
              "     다른 변수가 같은 정보를 이미 담고 있다는 뜻이다."]

    txt = "\n".join(L)
    with open(f"{OUT}/위험집단_요약.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print()
    print(txt)

    # ══════════════════════════════════════════════════════════
    log("")
    log("시각화 생성 중...")
    # ══════════════════════════════════════════════════════════
    # (1) 카드 이미지 — PPT용  (위험·안전 카드 수가 다를 수 있음)
    NCOL = max(len(top), len(bot), 1)
    fig, axes = plt.subplots(2, NCOL, figsize=(5.2 * NCOL, 7.4))
    axes = np.atleast_2d(axes)
    for row, (data, tone) in enumerate([(top, "위험"), (bot, "안전")]):
        for col in range(NCOL):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(data):
                continue
            r = data.iloc[col]
            face = "#FDECEA" if tone == "위험" else "#EAF6EE"
            edge = "#BF5620" if tone == "위험" else "#2C5F45"
            ax.add_patch(FancyBboxPatch(
                (0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.02",
                transform=ax.transAxes, facecolor=face, edgecolor=edge, lw=2))
            ax.text(0.5, 0.90, f"{tone} {col+1}위", transform=ax.transAxes,
                    ha="center", fontsize=12, color=edge, weight="bold")
            ax.text(0.5, 0.72, f"{r['부도율(%)']:.1f}%", transform=ax.transAxes,
                    ha="center", fontsize=40, color=edge, weight="bold")
            ax.text(0.5, 0.61, f"평균 {BASE*100:.2f}%의 "
                               f"{r['평균대비배수']:.2f}배",
                    transform=ax.transAxes, ha="center", fontsize=11,
                    color="#555555")
            desc = str(r["집단설명"]).replace("  ·  ", "\n")
            ax.text(0.5, 0.36, desc, transform=ax.transAxes, ha="center",
                    va="center", fontsize=11.5, color="#1A1A1A",
                    linespacing=1.7)
            ax.text(0.5, 0.11, f"{r['인원수']:,}명 · 전체의 {r['비율(%)']:.1f}%\n"
                               f"95% CI  {r['CI하한(%)']:.1f}% ~ {r['CI상한(%)']:.1f}%",
                    transform=ax.transAxes, ha="center", fontsize=9.5,
                    color="#6E6E6E", linespacing=1.5)
    fig.suptitle(f"부도 위험 집단 — 위험 {len(top)}개 / 안전 {len(bot)}개"
                 f"   (전체 {len(df):,}명 · 평균 {BASE*100:.2f}%)",
                 fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(f"{OUT}/위험집단_카드.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (2) 전체 막대
    fig, ax = plt.subplots(figsize=(14, max(5, len(res) * 0.52)))
    colors = plt.cm.RdYlGn_r(np.clip(res["평균대비배수"] / 3.5, 0, 1))
    ax.barh(range(len(res)), res["부도율(%)"], color=colors)
    ax.errorbar(res["부도율(%)"], range(len(res)),
                xerr=[res["부도율(%)"] - res["CI하한(%)"],
                      res["CI상한(%)"] - res["부도율(%)"]],
                fmt="none", ecolor="black", capsize=3, lw=0.9)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels([s[:58] for s in res["집단설명"]], fontsize=9)
    for i, r in res.iterrows():
        ax.text(r["CI상한(%)"] + 0.4, i,
                f"{r['부도율(%)']:.1f}%  ({r['인원수']:,}명)",
                va="center", fontsize=9)
    ax.axvline(BASE * 100, color="black", ls="--", lw=1.3,
               label=f"평균 {BASE*100:.2f}%")
    ax.invert_yaxis()
    ax.set_xlabel("부도율(%)")
    ax.set_xlim(0, res["CI상한(%)"].max() * 1.35)
    ax.legend()
    ax.set_title(f"SHAP 상위 변수 기반 위험 집단 — "
                 f"분별력 {res['부도율(%)'].max()/res['부도율(%)'].min():.1f}배")
    plt.tight_layout()
    plt.savefig(f"{OUT}/위험집단_그래프.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (3) 나무
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(clf, feature_names=list(X.columns), class_names=["정상", "부도"],
              filled=True, rounded=True, proportion=True, impurity=False,
              fontsize=8, ax=ax)
    ax.set_title(f"위험 집단 나무 (깊이 {DEPTH}) — 색이 진할수록 부도율 높음",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT}/위험집단_나무.png", dpi=130, bbox_inches="tight")
    plt.close()

    df["집단"] = df["_leaf"].map(dict(zip(res["_leaf"], res["순위"])))
    df[band_cols + ["집단", TARGET]].to_pickle(f"{OUT}/위험집단_할당.pkl")

    log("")
    log("저장: 위험집단_등급기준표.csv / 위험집단_규칙.csv / 위험집단_카드.csv")
    log("      위험집단_요약.txt / 위험집단_카드.png")
    log("      위험집단_그래프.png / 위험집단_나무.png")
    log(f"소요시간 {time.time()-t0:.1f}초")
