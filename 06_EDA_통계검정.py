# -*- coding: utf-8 -*-
"""
STEP 5. EDA 및 통계검정
- 목적: 어떤 변수가 실제로 목표변수(부도여부)와 통계적으로 유의미하게 관련 있는지 확인
- 입력: 04b_결측치처리_유형별가이드_v2.py 가 만든 full_feature_table_imputed_v2.pkl
       (04 vs 04b 비교 결과 04b가 속도·해석력 면에서 더 유리해 최종 채택함.
        03번은 더 이상 쓰지 않고, 04b번이 마스터 CSV를 바로 읽어서 처리한 결과를
        05b/06/07번이 이어받는 구조로 통일함)
- 출력: 수치형/범주형 변수 통계검정 결과표(csv), 상관관계 히트맵/박스플롯/부도율 바차트(png)

기법 선택 이유:
  - 수치형 변수 vs 목표변수 : Mann-Whitney U 검정
      -> 소득·대출금액처럼 오른쪽으로 크게 치우친(right-skewed) 분포가 많아서,
         평균 차이를 보는 t-test보다 순위(중앙값) 차이를 보는 비모수 검정이 더 적합함
  - 범주형 변수 vs 목표변수 : 카이제곱(Chi-square) 독립성 검정
      -> "이 범주형 변수와 부도여부가 서로 독립인가"를 직접 검정하는 표준적인 방법
  - 수치형 변수 간 상관관계 : Spearman 상관계수
      -> 이상치·비선형 관계에 덜 민감해서, 왜곡된 금융 데이터에 Pearson보다 안전함
"""
import time
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency

# 한글 폰트 설정 (macOS 기본 폰트 사용, 없으면 자동 무시)
for _font in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if _font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False

UP = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋"
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def print_progress(current, total, label="진행", every=5):
    """[진행] |#####-----| 12/183 ( 6.6%) 형태의 막대를 콘솔에 출력.
    처음에는 \\r(캐리지리턴)로 같은 줄을 계속 덮어쓰는 방식이었는데, Spyder의 IPython
    콘솔(+ 최신 python3.14 free-threaded 빌드 조합)에서는 \\r이 그때그때 화면에 반영되지
    않고 버퍼에 쌓였다가 한꺼번에 나오는 경우가 있어서, 실제로는 잘 돌아가고 있는데도
    "바가 안 뜨고 멈춘 것처럼" 보이는 문제가 있었다. 그래서 매번 덮어쓰지 않고, 5%(또는
    처음/마지막)마다 새 줄로 출력하는 방식으로 바꿔 어떤 콘솔 환경에서도 확실히 보이게 한다."""
    pct = current / total * 100
    step = max(1, total // (100 // every))  # 대략 every%마다 한 번 출력
    if current == 1 or current == total or current % step == 0:
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"[{label}] |{bar}| {current}/{total} ({pct:5.1f}%)", flush=True)


def load_train():
    df = pd.read_pickle(f"{OUT}/full_feature_table_imputed_v2.pkl")
    train = df[df["IS_TRAIN"] == 1].copy()
    del df  # test 행까지 포함된 원본(약 550MB)을 더 들고 있을 필요 없으므로 즉시 해제
    gc.collect()
    train[TARGET] = train[TARGET].astype(int)
    log(f"학습대상(train) 로드: {train.shape}, 부도율={train[TARGET].mean():.4f}")
    return train


# ---------------------------------------------------------------------------
# 0) 수치형/범주형 컬럼 재분류
#    - dtype이 숫자(int/float)여도 실제 값이 0/1 두 가지뿐인 컬럼(서류제출여부_*,
#      휴대폰제공여부 등 FLAG류, 등록실거주지역불일치 등 불일치 플래그, 04b에서 만든
#      HAS_BUREAU/HAS_BUILDING_INFO/외부신용점수*_MISSING/무직_또는_연금수급여부/차량정보있음 등)는
#      "숫자로 저장된 범주형"이므로 Mann-Whitney U가 아니라 카이제곱 검정 대상으로 분류한다.
#      -> 판단 기준: 결측을 제외한 고유값 개수가 2개 이하이면 이진 범주형으로 간주.
#    - 반대로 "숫자가 아니면 무조건 범주형"으로 먼저 걸러낸다. is_object_dtype만으로는
#      pyarrow 백엔드 문자열(string[pyarrow])이나 category dtype 컬럼을 못 잡아서
#      숫자형으로 잘못 분류되고, 이후 np.nanstd/mean 계산에서
#      "TypeError: unsupported operand type(s) for /: 'str' and 'int'" 에러가 났었다.
#      is_numeric_dtype으로 먼저 숫자인지 확인하고, 숫자가 아니면 dtype 종류와 상관없이
#      바로 범주형으로 분류하도록 순서를 뒤집어 이 문제를 근본적으로 막는다.
# ---------------------------------------------------------------------------
def classify_columns(train: pd.DataFrame, exclude=("고유ID", TARGET, "IS_TRAIN")):
    numeric_cols, categorical_cols = [], []
    for c in train.columns:
        if c in exclude:
            continue
        s = train[c]
        if pd.api.types.is_bool_dtype(s):
            categorical_cols.append(c)
        elif not pd.api.types.is_numeric_dtype(s):
            # object, category, string, string[pyarrow] 등 숫자가 아닌 모든 dtype -> 범주형
            categorical_cols.append(c)
        elif s.nunique(dropna=True) <= 2:
            categorical_cols.append(c)   # 0/1 이진 플래그 -> 범주형으로 재분류
        else:
            numeric_cols.append(c)
    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# 1) 수치형 변수 vs 목표변수 : Mann-Whitney U 검정
# ---------------------------------------------------------------------------
def test_numeric_vs_target(train: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    # 주의: g0 = train[train[TARGET]==0] 처럼 전체 컬럼을 통째로 미리 복사해두면
    # 컬럼 1개만 필요한데도 300~500MB짜리 데이터프레임 사본을 계속 들고 있게 되어
    # 메모리가 부족한 컴퓨터에서 커널이 죽는 원인이 된다. 마스크만 미리 만들고
    # 컬럼은 매 반복마다 하나씩만 꺼내 쓰는 방식으로 메모리 사용량을 최소화한다.
    is_bad = train[TARGET].values == 1
    is_good = ~is_bad

    rows = []
    total = len(numeric_cols)
    log(f"수치형 변수 검정 시작 (총 {total}개, 컬럼당 30만행 순위계산이라 몇 분 걸릴 수 있음)")
    for i, c in enumerate(numeric_cols, start=1):
        print_progress(i, total, "수치형 검정")
        col = train[c].values
        x0, x1 = col[is_good], col[is_bad]
        if np.nanstd(x0) == 0 and np.nanstd(x1) == 0:
            continue
        try:
            stat, p = mannwhitneyu(x0, x1, alternative="two-sided")
        except ValueError:
            continue
        # rank-biserial 상관계수 (효과크기, U를 -1~1로 정규화)
        n0, n1 = len(x0), len(x1)
        effect = 1 - (2 * stat) / (n0 * n1)
        rows.append({
            "컬럼명": c,
            "정상_평균": np.nanmean(x0), "부도_평균": np.nanmean(x1),
            "정상_중앙값": np.nanmedian(x0), "부도_중앙값": np.nanmedian(x1),
            "U통계량": stat, "p_value": p,
            "유의(0.05)": p < 0.05,
            "효과크기(rank-biserial)": effect,
        })
    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 2) 범주형 변수 vs 목표변수 : 카이제곱 독립성 검정
# ---------------------------------------------------------------------------
def test_categorical_vs_target(train: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    rows = []
    total = len(cat_cols)
    log(f"범주형 변수 검정 시작 (총 {total}개)")
    for i, c in enumerate(cat_cols, start=1):
        print_progress(i, total, "범주형 검정")
        ct = pd.crosstab(train[c], train[TARGET])
        if ct.shape[0] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        rate_by_cat = train.groupby(c)[TARGET].mean()
        rows.append({
            "컬럼명": c,
            "카이제곱통계량": chi2, "자유도": dof, "p_value": p,
            "유의(0.05)": p < 0.05,
            "부도율_최대": rate_by_cat.max(), "부도율_최소": rate_by_cat.min(),
            "부도율_최대범주": rate_by_cat.idxmax(), "부도율_최소범주": rate_by_cat.idxmin(),
        })
    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 3) 상관관계 히트맵 (목표변수와 상관 높은 상위 30개, Spearman)
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(train: pd.DataFrame, numeric_result: pd.DataFrame, top_n=30, path=None):
    top_cols = numeric_result.sort_values("효과크기(rank-biserial)", key=abs, ascending=False).head(top_n)["컬럼명"].tolist()
    corr = train[top_cols + [TARGET]].corr(method="spearman")

    plt.figure(figsize=(14, 12))
    im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("주요 변수 상관관계 히트맵 (Spearman, 목표변수 영향력 상위 30개)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 4) 주요 수치형 변수 박스플롯 (목표변수별 분포 비교)
# ---------------------------------------------------------------------------
def plot_top_numeric_boxplots(train: pd.DataFrame, numeric_result: pd.DataFrame, top_n=6, path=None):
    top_cols = numeric_result.head(top_n)["컬럼명"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, c in zip(axes.ravel(), top_cols):
        data = [train.loc[train[TARGET] == 0, c], train.loc[train[TARGET] == 1, c]]
        # matplotlib 버전에 따라 boxplot()의 labels/tick_labels 파라미터 이름이 달라서
        # (3.9 이전=labels, 3.11+=labels 제거·tick_labels로 대체) 버전에 안 흔들리게
        # set_xticklabels로 따로 지정
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(["정상(0)", "부도(1)"])
        ax.set_title(c, fontsize=10)
    plt.suptitle("목표변수별 분포 비교 - p-value 상위 6개 변수 (이상치 제외 표시)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5) 주요 범주형 변수 부도율 바차트
# ---------------------------------------------------------------------------
def plot_top_categorical_bars(train: pd.DataFrame, cat_result: pd.DataFrame, top_n=4, path=None):
    top_cols = cat_result.head(top_n)["컬럼명"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    overall_rate = train[TARGET].mean()
    for ax, c in zip(axes.ravel(), top_cols):
        rate = train.groupby(c)[TARGET].mean().sort_values(ascending=False)
        ax.bar(rate.index.astype(str), rate.values)
        ax.axhline(overall_rate, color="red", linestyle="--", linewidth=1, label="전체 평균 부도율")
        ax.set_title(c, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(fontsize=8)
    plt.suptitle("범주별 부도율 - p-value 상위 4개 변수")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    t0 = time.time()
    train = load_train()

    numeric_cols, categorical_cols = classify_columns(train)
    log(f"컬럼 재분류 완료 -> 수치형 {len(numeric_cols)}개 / 범주형(문자+이진플래그) {len(categorical_cols)}개")

    numeric_result = test_numeric_vs_target(train, numeric_cols)
    numeric_result.to_csv(f"{OUT}/EDA_수치형변수_통계검정.csv", index=False, encoding=ENC)
    log(f"수치형 변수 검정 완료: {len(numeric_result)}개 중 유의 {numeric_result['유의(0.05)'].sum()}개")

    cat_result = test_categorical_vs_target(train, categorical_cols)
    cat_result.to_csv(f"{OUT}/EDA_범주형변수_통계검정.csv", index=False, encoding=ENC)
    log(f"범주형 변수 검정 완료: {len(cat_result)}개 중 유의 {cat_result['유의(0.05)'].sum()}개")

    plot_correlation_heatmap(train, numeric_result, path=f"{OUT}/EDA_상관관계_히트맵.png")
    plot_top_numeric_boxplots(train, numeric_result, path=f"{OUT}/EDA_주요변수_박스플롯.png")
    plot_top_categorical_bars(train, cat_result, path=f"{OUT}/EDA_범주형_부도율_바차트.png")
    log("시각화 3종 저장 완료")

    log(f"전체 소요시간: {time.time()-t0:.1f}초")
