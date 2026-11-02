# -*- coding: utf-8 -*-
"""
STEP 25. 통합 스코어링 함수 (파이프라인 통합 2/2)

배경:
- 24번이 저장한 "최종_스코어링_아티팩트.pkl" 하나만 있으면, 이 파일의 score_customer()
  함수 호출 한 번으로 새 고객 1명의 A~E 등급 + 권장조치까지 바로 나온다. 24번을 다시
  실행할 필요 없이(=XGBoost 재학습 없이) 이 파일만 import해서 쓰면 된다.

입력 스키마(중요): score_customer()에 넣는 고객 데이터는 "03번(다중테이블 병합) +
04b번(결측치 처리) + 07번(파생변수 생성)"까지 이미 끝난 상태여야 한다. 즉
full_feature_table_derived.pkl과 같은 컬럼 구성(대출금액, 총소득, 외부신용점수_평균,
성별, 최종학력 등 원본+파생변수 전부)의 딕셔너리 1개(고객 1명)를 넣어야 한다.
이 스크립트는 그 뒤 단계(08 인코딩 -> 12 모델 -> 17 보정 -> 18 등급 -> 23 의사결정)만
책임진다 - 원시 신청서/이력 테이블(bureau.csv 등)을 직접 받아 처리하지는 않는다
(그 부분은 이미 만들어진 03/04b/07번 스크립트를 그대로 쓰면 됨).

사용 예시:
    from importlib import import_module
    import sys
    sys.path.append("/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/코드")
    scorer = import_module("25_homecredit_scorer")  # 파일명이 숫자로 시작해서 이렇게 로드
    result = scorer.score_customer(고객딕셔너리)
    print(result)

    (또는 이 파일 맨 아래 데모 코드처럼 이 스크립트를 그대로 실행해도 됨)
"""
import pickle
import numpy as np
import pandas as pd

OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ARTIFACT_PATH = f"{OUT}/최종_스코어링_아티팩트.pkl"

_artifact = None  # 모듈 안에서 한 번만 로드해서 재사용 (매번 파일 읽지 않도록 캐싱)


def _load_artifact():
    global _artifact
    if _artifact is None:
        with open(ARTIFACT_PATH, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact


def _to_logit(p, eps):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).reshape(-1, 1)


def score_customer(row):
    """
    고객 1명을 채점해서 등급/권장조치를 반환한다.

    Parameters
    ----------
    row : dict 또는 pandas.Series
        full_feature_table_derived.pkl과 같은 컬럼 구성을 가진 고객 1명의 데이터.
        artifact["feat_cols"]에 있는 컬럼이 전부 존재할 필요는 없음 - 없는 값은
        NaN으로 간주하고(수치형은 학습 시점 중앙값으로, 범주형은 "MISSING"으로) 처리함.

    Returns
    -------
    dict : {
        "원시예측확률": XGBoost가 직접 뱉은 확률(보정 전, 실제 부도율과는 괴리 있음),
        "보정확률": Platt 보정 후 확률(실제 부도율에 가까운 진짜 확률),
        "등급": "A"~"E",
        "권장조치": 예) "승인+우대금리" / "표준승인" / "조건부승인(...)" / "거절",
        "위험조정가산금리(%p)": 이 등급에 적용할 최소 가산금리,
        "등급별_참고_실제부도율": 이 등급에 속한 사람들의 실측 평균 부도율(학습 시점 기준),
    }
    """
    art = _load_artifact()

    if isinstance(row, dict):
        row = pd.Series(row)
    X = pd.DataFrame([row])

    # 1) 범주형 라벨인코딩 (24번이 저장한 매핑표 그대로 적용, 없는 카테고리는 NaN -> XGBoost가
    #    자체적으로 결측치로 처리함)
    for c in art["char_cat_cols"]:
        val = X[c].iloc[0] if c in X.columns else np.nan
        val = "MISSING" if pd.isna(val) else val
        X[c] = [art["label_maps"][c].get(val, np.nan)]

    # 2) 학습 때 쓴 컬럼 순서로 재정렬, 없는 컬럼은 NaN으로 채움
    X = X.reindex(columns=art["feat_cols"])

    # 3) 수치형 결측치는 학습 시점 중앙값으로 채움 (범주형/이진플래그는 XGBoost가 NaN을
    #    자체 처리하므로 그대로 둠)
    X[art["numeric_cols"]] = X[art["numeric_cols"]].fillna(art["median_fill"])

    # 4) 원시 확률 -> Platt 보정 -> 등급 -> 의사결정
    p_raw = art["xgb_model"].predict_proba(X)[:, 1][0]
    p_calib = art["platt_model"].predict_proba(_to_logit(np.array([p_raw]), art["platt_eps"]))[:, 1][0]
    grade = pd.cut([p_calib], bins=art["grade_bins"], labels=art["grades"])[0]
    dec = art["decision_table"][grade]

    return {
        "원시예측확률": round(float(p_raw), 4),
        "보정확률": round(float(p_calib), 4),
        "등급": str(grade),
        "권장조치": dec["권장조치"],
        "위험조정가산금리(%p)": round(dec["위험조정가산금리(%p)"], 2),
        "등급별_참고_실제부도율": round(dec["실제부도율"], 4),
    }


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    """여러 고객을 한 번에 채점하고 싶을 때 사용 (내부적으로 score_customer를 행마다 호출)."""
    results = [score_customer(row) for _, row in df.iterrows()]
    return pd.DataFrame(results)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 데모: full_feature_table_derived.pkl에서 무작위 고객 3명을 뽑아 실제로 채점해본다
    # ------------------------------------------------------------------
    demo = pd.read_pickle(f"{OUT}/full_feature_table_derived.pkl")
    demo = demo[demo["IS_TRAIN"] == 1].sample(3, random_state=1)
    print("=== 데모: 무작위 고객 3명 채점 결과 ===")
    for i, (_, r) in enumerate(demo.iterrows(), 1):
        result = score_customer(r)
        actual = r.get("목표변수_상환어려움여부", "알수없음")
        print(f"\n[고객 {i}] 실제 결과(참고용)={actual}")
        for k, v in result.items():
            print(f"  {k}: {v}")
