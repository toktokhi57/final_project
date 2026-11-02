# -*- coding: utf-8 -*-
"""
STEP 16. SHAP 모델 해석
- 목적: 최종 모델(XGBoost)의 예측 근거를 설명 가능하게 만든다.
  - Summary plot(전역 해석): 검증셋 전체적으로 어떤 변수가 위험도 판단에 가장 큰 영향을
    주는지, 그 변수가 높을 때/낮을 때 위험도가 어느 방향으로 움직이는지를 한눈에 보여준다.
  - Force/Waterfall plot(개별 해석): 특정 고객 한 명의 예측이 "왜" 그렇게 나왔는지,
    어떤 변수가 위험도를 얼마나 밀어올리고/끌어내렸는지를 분해해서 보여준다.

모델: XGBoost. 12번에서 확정한 기본 파라미터 그대로 사용(13번에서 튜닝해도 성능 차이가
      없었으므로). 리샘플링(RandomUnderSampler 등)은 최종 파이프라인에서 채택된 적이
      없고 08/12~15번 전부 scale_pos_weight만으로 불균형을 보정했으므로 여기서도 동일하게
      scale_pos_weight만 사용한다(이중보정 방지).

절차:
  1) valid셋에서 SHAP_SAMPLE_SIZE(기본 2,000)개를 무작위 샘플링
     (TreeSHAP이 KernelSHAP보다 훨씬 빠르지만, 6만여 행 전체를 다 그리면 summary plot이
     너무 빽빽해지고 느려지므로 샘플링해서 사용)
  2) Summary plot 생성 -> 06번(EDA)/07번(파생변수) 결과와 상위 변수가 겹치는지 눈으로
     교차검증할 수 있도록 상위 20개 변수 목록도 별도로 저장
  3) 15번에서 확정한 임계값(0.665) 기준으로 valid셋에서 고위험/저위험/경계 사례를 각 1명씩
     선정 (예측확률이 가장 높은 사람 / 가장 낮은 사람 / 0.665에 가장 가까운 사람)
  4) 그 3명에 대해 Waterfall plot 생성 -> STEP18(위험등급 부여) 단계에서 "왜 이 등급인지"
     설명할 때 그대로 재사용 가능

필요 패키지: shap (없으면 설치 필요 - conda install -c conda-forge shap 또는 pip install shap)

입력: tree_encoded_train/valid.pkl (08번 출력)
출력: SHAP_summary_plot.png, SHAP_상위변수_TOP20.csv,
      SHAP_waterfall_고위험.png, SHAP_waterfall_저위험.png, SHAP_waterfall_경계.png
"""
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
# AppleGothic만 쓰면 한글은 나오지만 유니코드 마이너스 기호(−, U+2212)가 없어서
# 음수 값(근속일수/나이 등 원본이 음수로 저장된 컬럼)이 네모(tofu)로 깨짐.
# sans-serif 폰트 목록으로 fallback을 걸어서, 한글은 AppleGothic이 담당하고
# AppleGothic에 없는 글리프(마이너스 기호 등)는 뒤 순서 폰트가 대신 그리게 함
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import shap
except ImportError as e:
    raise ImportError(
        "shap 패키지가 필요합니다. 터미널에서 'conda activate homecredit' 후 "
        "'conda install -c conda-forge shap -y' (또는 'pip install shap') "
        "실행하고 Spyder 커널을 재시작한 뒤 다시 시도하세요."
    ) from e

# shap가 아직 XGBoost 3.0+의 base_score 저장 형식("[5E-1]" 같은 리스트 문자열)을 못 읽어서
# TreeExplainer 생성 시 ValueError가 난다 (shap GitHub Issue #4202, 아직 미해결).
# XGBoost 모델 쪽 설정을 고쳐도 save_config() 호출 시 다시 원래 형식으로 재직렬화되어
# 무효화되므로, shap._tree 모듈 안에서 float() 호출만 대괄호를 벗겨내도록 패치해서 우회한다.
import shap.explainers._tree as _shap_tree_mod

_real_float = float


def _safe_float(x):
    if isinstance(x, str):
        x = x.strip().strip("[]")
    return _real_float(x)


_shap_tree_mod.float = _safe_float

# waterfall plot의 라벨 문자열(예: "-4268 = 근속후경과일", "+0.94")은 shap.plots._waterfall
# 모듈 안의 format_value() 함수가 만드는데, 이 함수가 내부적으로 '-'를 유니코드 마이너스
# (−, U+2212)로 바꿔서 반환한다. 그림을 그린 뒤에 텍스트를 고치는 방식은 shap이 레이아웃을
# 다시 계산할 때(tight_layout/savefig) 라벨을 다시 만들면서 원상복구돼 효과가 없었다.
# 그래서 라벨을 "만드는" 함수 자체를 패치해서 애초에 문제되는 문자가 안 나오게 막는다.
import re
import shap.plots._waterfall as _shap_waterfall_mod

_DASH_PATTERN_EARLY = re.compile("[‐‑‒–—―−]")
_real_format_value = _shap_waterfall_mod.format_value


def _ascii_format_value(*args, **kwargs):
    s = _real_format_value(*args, **kwargs)
    return _DASH_PATTERN_EARLY.sub("-", s)


_shap_waterfall_mod.format_value = _ascii_format_value

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"
RANDOM_STATE = 42

FINAL_THRESHOLD = 0.665  # 15/15_1번에서 확정한 최종 임계값
SHAP_SAMPLE_SIZE = 2000  # summary plot용 샘플 크기 (1,000~3,000 권장)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


import re

# 유니코드 대시/마이너스 계열 문자 모음 (shap/numpy가 상황에 따라 다른 걸 씀) - 전부 일반 하이픈으로 통일
_DASH_PATTERN = re.compile("[‐‑‒–—―−]")


def fix_unicode_minus(fig=None):
    """shap이 숫자 앞 '-'를 여러 종류의 유니코드 대시/마이너스 문자로 바꿔서 그리는데,
    이 글리프가 없는 폰트(AppleGothic 등)에서는 깨져 보인다. 폰트로 해결하는 대신,
    그려진 텍스트 객체를 직접 찾아서 전부 일반 하이픈(-)으로 바꿔치기해 확실하게 고친다."""
    fig = fig or plt.gcf()
    fig.canvas.draw()  # 레이아웃/텍스트가 완전히 확정된 뒤에 스캔하도록 강제로 한 번 그림
    for t in fig.findobj(matplotlib.text.Text):
        txt = t.get_text()
        new_txt = _DASH_PATTERN.sub("-", txt)
        if new_txt != txt:
            t.set_text(new_txt)
        # 진단용: 아스키(영문/숫자/기본기호) 범위를 벗어나는 문자가 남아있으면
        # 정확히 어떤 문자(코드포인트)인지 콘솔에 출력 - 다음 원인 파악용
        remaining = [c for c in t.get_text() if ord(c) > 127 and not ("가" <= c <= "힣")]
        if remaining:
            codes = ", ".join(f"{c!r}(U+{ord(c):04X})" for c in remaining)
            print(f"[진단] 미확인 특수문자 발견 -> 텍스트='{t.get_text()}' / 문자={codes}")


if __name__ == "__main__":
    t0 = time.time()

    # =====================================================================
    # 데이터 로드 및 XGBoost 학습 (12/13/14/15번과 동일 파라미터, 리샘플링 없음)
    # =====================================================================
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
    log(f"XGBoost 학습 완료 (train {X_tr.shape}, valid {X_val.shape})")

    # =====================================================================
    # 1) SHAP 값 계산 (TreeSHAP, valid셋에서 샘플링)
    # =====================================================================
    log("=" * 70)
    log(f"SHAP 계산 시작 (valid셋에서 {SHAP_SAMPLE_SIZE}개 샘플링)")
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_val), size=min(SHAP_SAMPLE_SIZE, len(X_val)), replace=False)
    X_sample = X_val.iloc[sample_idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):  # shap 버전에 따라 클래스별 리스트로 나올 수 있음
        shap_values = shap_values[1]
    log(f"SHAP 값 계산 완료: shape={np.array(shap_values).shape}")

    # =====================================================================
    # 2) Summary plot (전역 해석)
    # =====================================================================
    log("=" * 70)
    log("Summary plot 생성 중...")
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    fix_unicode_minus()
    plt.savefig(f"{OUT}/SHAP_summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top20 = (
        pd.DataFrame({"변수": X_sample.columns, "평균절대SHAP값": mean_abs_shap})
        .sort_values("평균절대SHAP값", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top20.to_csv(f"{OUT}/SHAP_상위변수_TOP20.csv", index=False, encoding=ENC)
    log("=== SHAP 상위 20개 변수 (06/07번 EDA·파생변수 결과와 교차검증용) ===")
    print(top20.to_string(index=False))

    # =====================================================================
    # 3) 임계값(0.665) 기준 고위험/저위험/경계/고위험적중 사례 선정
    # =====================================================================
    log("=" * 70)
    idx_high = int(np.argmax(p_val))
    idx_low = int(np.argmin(p_val))
    idx_border = int(np.argmin(np.abs(p_val - FINAL_THRESHOLD)))
    # 고위험(TP): 실제 부도자(y_val==1) 중 예측확률이 가장 높은 사람 -> 모델이 정확히 맞춘
    # "확신에 찬 위험 예측" 사례. 위 idx_high(예측확률 최고 전체)는 결과적으로 실제 정상이라
    # False Positive였는데, 이 사례와 나란히 비교하면 "맞춘 위험 예측 vs 틀린 위험 예측"을
    # 보여줄 수 있다.
    default_idx = np.where(y_val == 1)[0]
    idx_tp = int(default_idx[np.argmax(p_val[default_idx])])

    log(f"고위험 사례(전체 최고확률): valid행 {idx_high}, 예측확률={p_val[idx_high]:.4f}, "
        f"실제={y_val[idx_high]} ({'적중' if y_val[idx_high]==1 else '오판(FP)'})")
    log(f"고위험적중 사례(TP, 실제 부도자 중 최고확률): valid행 {idx_tp}, "
        f"예측확률={p_val[idx_tp]:.4f}, 실제={y_val[idx_tp]}")
    log(f"저위험 사례: valid행 {idx_low}, 예측확률={p_val[idx_low]:.4f}, 실제={y_val[idx_low]}")
    log(f"경계 사례: valid행 {idx_border}, 예측확률={p_val[idx_border]:.4f}, 실제={y_val[idx_border]}")

    # =====================================================================
    # 4) Waterfall plot (개별 해석) - 위 4명 각각
    # =====================================================================
    log("=" * 70)
    log("Waterfall plot 생성 중 (고위험/고위험적중TP/저위험/경계)...")
    for name, idx in [("고위험", idx_high), ("고위험적중_TP", idx_tp),
                       ("저위험", idx_low), ("경계", idx_border)]:
        row = X_val.iloc[[idx]]
        sv = explainer(row)  # shap.Explanation 객체 (최신 API)
        if len(sv.shape) == 3:  # (n, features, n_class) 형태로 나오는 버전 대응
            sv = sv[:, :, 1]
        plt.figure()
        shap.plots.waterfall(sv[0], show=False, max_display=15)
        plt.title(f"{name} 사례 (예측확률={p_val[idx]:.4f}, 실제={'부도' if y_val[idx]==1 else '정상'})")
        plt.tight_layout()
        fix_unicode_minus()
        plt.savefig(f"{OUT}/SHAP_waterfall_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        log(f"[{name}] 저장 완료 (SHAP_waterfall_{name}.png)")

    log("=" * 70)
    log(f"전체 소요시간: {time.time()-t0:.1f}초")
