# -*- coding: utf-8 -*-
"""
STEP 6. 파생변수 생성
- 목적: 06_EDA_통계검정.py 에서 확인된 핵심 신호(외부신용점수, 소득 대비 대출부담,
  근속·나이 비율, 주소/근무지 정보 불일치, 서류제출·신용조회 이력 등)를 원 raw 컬럼들의
  "비율"이나 "합계"로 재조합해서, 로지스틱 스코어카드와 트리모델(RF/XGB/LGBM) 둘 다
  더 쉽게 신호를 잡아낼 수 있게 만든다.
  - 로지스틱 회귀는 컬럼 간 곱셈/나눗셈 관계를 스스로 못 찾기 때문에 비율 피처를
    직접 만들어줘야 성능이 오른다.
  - 트리모델은 원칙적으로 원본 컬럼만으로도 비슷한 분기를 찾을 수 있지만, 비율을
    직접 주면 분기 횟수가 줄어들어 더 안정적으로 같은 패턴을 학습한다.
- 입력: 04b_결측치처리_유형별가이드_v2.py 가 만든 마스터 파일 full_feature_table_imputed_v2.pkl
  (04 vs 04b 비교 결과 04b를 최종 채택했으므로 이 파일을 기본 입력으로 사용함)
- 출력: full_feature_table_derived.pkl / .csv
  (05b_모델비교_파생변수포함.py, 08_확률보정_등급변환 등 이후 단계는 이 파일을 새 마스터로 읽으면 됨)

파생변수 설계 근거 (전부 06번 EDA 결과 또는 Home Credit 대회에서 잘 알려진 신호 기반):
  1) 신용부담비율_대출소득 = 대출금액 / 총소득
     -> 대출 절대금액보다 "소득 대비 부담 정도"가 부도와 더 직접적으로 연결됨
  2) 신용부담비율_상환소득 = 연간상환액 / 총소득
     -> 매년 갚아야 하는 금액이 소득에서 차지하는 비중 (DTI 성격 지표)
  3) 신용부담비율_상환대출 = 연간상환액 / 대출금액
     -> 사실상 대출기간의 역수 성격. 짧은 기간에 많이 갚아야 하면 부담이 큼
  4) 계약금비율 = 구매물품가격 / 대출금액
     -> 실제 물건값 대비 얼마나 빌렸는지 (자기부담금 비율의 반대 개념)
  5) 근속연령비율 = 근속후경과일 / 출생후경과일
     -> 06번에서 두 변수 모두 개별 효과크기가 컸는데, 이 둘의 비율(나이 대비 근속기간)이
        Home Credit 대회에서 개별 변수보다 더 강력한 신호로 잘 알려져 있음
  6~9) 외부신용점수_평균 / 최솟값 / 표준편차 / 결측개수
     -> 06번에서 외부신용점수1/2/3이 개별적으로 가장 강한 변수였음. 세 점수를 종합한
        평균·최솟값과, 몇 개나 결측인지(정보 부족 자체가 위험신호)를 추가
  10) 가구원당소득 = 총소득 / 가족구성원수
     -> 총소득이 같아도 부양가족 수에 따라 실질 여유자금이 다름
  11) 정보불일치_총합 = 등록/실거주/근무지 불일치 플래그 5개의 합 (0~5점)
     -> 06번에서 5개 불일치 플래그가 전부 개별적으로 유의했음. 하나로 합쳐
        "정보 신뢰도" 지표로 요약 (높을수록 위험)
  12) 서류제출_총건수 = 서류제출여부_1~21번 합
     -> Home Credit 대회에서 잘 알려진 FLAG_DOCUMENT 합계 피처. 제출 서류 수 자체가
        신청 성실도/심사 강도를 반영
  13) 최근신용조회_총합 = 신용조회건수(1시간전~1년전) 합
     -> 06번에서 개별 조회건수 변수들이 유의했음. 총 조회 횟수로 "최근 신용 활동 강도" 요약
  14) BUREAU_활성대출비율 = BUREAU_IS_ACTIVE_SUM / BUREAU_신용정보국대출ID_COUNT
     -> 전체 대출 이력 대비 현재 활성 대출 비중 (부채 관리 상태)
  15) PREV_거절이력여부 = (PREV_IS_REFUSED_SUM > 0)
     -> 06번에서 PREV_IS_REFUSED_SUM(과거 거절 횟수)이 강한 신호였음. 이진 플래그로도 추가
  16) 연체이력존재여부 = (INST_IS_LATE_MEAN > 0)
     -> INST_IS_LATE_MEAN(할부 연체 비율)이 강한 신호였음. "한 번이라도 연체했는지"의
        이진 버전을 추가해 트리모델이 분기점을 더 쉽게 찾게 함
"""
import time
import gc
import numpy as np
import pandas as pd

UP = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋"
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"
TARGET = "목표변수_상환어려움여부"

SRC_PKL = f"{OUT}/full_feature_table_imputed_v2.pkl"   # 04b번 출력(최종 채택된 마스터)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include="int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include="float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """분모가 0이거나 결측이면 NaN을 반환 (0으로 나누기 방지)."""
    denom_safe = denom.replace(0, np.nan)
    return numer / denom_safe


# ---------------------------------------------------------------------------
# 파생변수 생성 본체
# ---------------------------------------------------------------------------
def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    n_before = df.shape[1]

    # 1~4) 소득/대출 부담 비율
    df["신용부담비율_대출소득"] = safe_div(df["대출금액"], df["총소득"])
    df["신용부담비율_상환소득"] = safe_div(df["연간상환액"], df["총소득"])
    df["신용부담비율_상환대출"] = safe_div(df["연간상환액"], df["대출금액"])
    df["계약금비율"] = safe_div(df["구매물품가격"], df["대출금액"])

    # 5) 근속연령비율 (둘 다 음수 DAYS 값이라 비율 자체는 양수로 나옴)
    df["근속연령비율"] = safe_div(df["근속후경과일"], df["출생후경과일"])

    # 6~9) 외부신용점수 종합
    ext_cols = ["외부신용점수1", "외부신용점수2", "외부신용점수3"]
    ext_cols = [c for c in ext_cols if c in df.columns]
    if ext_cols:
        ext = df[ext_cols]
        df["외부신용점수_평균"] = ext.mean(axis=1)
        df["외부신용점수_최솟값"] = ext.min(axis=1)
        df["외부신용점수_표준편차"] = ext.std(axis=1)
        df["외부신용점수_결측개수"] = ext.isna().sum(axis=1)

    # 10) 가구원당소득
    df["가구원당소득"] = safe_div(df["총소득"], df["가족구성원수"])

    # 11) 정보 불일치 총합 (0~5점)
    mismatch_cols = [
        "등록근무도시불일치", "등록실거주도시불일치", "실거주근무도시불일치",
        "등록근무지역불일치", "등록실거주지역불일치",
    ]
    mismatch_cols = [c for c in mismatch_cols if c in df.columns]
    if mismatch_cols:
        df["정보불일치_총합"] = df[mismatch_cols].sum(axis=1)

    # 12) 서류제출 총건수 (서류제출여부_1번~21번)
    doc_cols = [c for c in df.columns if c.startswith("서류제출여부_")]
    if doc_cols:
        df["서류제출_총건수"] = df[doc_cols].sum(axis=1)

    # 13) 최근 신용조회 총합
    inquiry_cols = [
        "신용조회건수_1시간전", "신용조회건수_1일전", "신용조회건수_1주전",
        "신용조회건수_1개월전", "신용조회건수_1분기전", "신용조회건수_1년전",
    ]
    inquiry_cols = [c for c in inquiry_cols if c in df.columns]
    if inquiry_cols:
        df["최근신용조회_총합"] = df[inquiry_cols].sum(axis=1)

    # 14) BUREAU 활성대출비율
    if "BUREAU_IS_ACTIVE_SUM" in df.columns and "BUREAU_신용정보국대출ID_COUNT" in df.columns:
        df["BUREAU_활성대출비율"] = safe_div(df["BUREAU_IS_ACTIVE_SUM"], df["BUREAU_신용정보국대출ID_COUNT"])

    # 15) 과거 거절 이력 여부 (이진 플래그)
    if "PREV_IS_REFUSED_SUM" in df.columns:
        df["PREV_거절이력여부"] = (df["PREV_IS_REFUSED_SUM"] > 0).astype("int8")

    # 16) 연체 이력 존재 여부 (이진 플래그)
    if "INST_IS_LATE_MEAN" in df.columns:
        df["연체이력존재여부"] = (df["INST_IS_LATE_MEAN"] > 0).astype("int8")

    n_after = df.shape[1]
    log(f"파생변수 {n_after - n_before}개 생성 완료 -> {df.shape}")
    return df


if __name__ == "__main__":
    t0 = time.time()

    df = pd.read_pickle(SRC_PKL)
    df = downcast(df)
    log(f"마스터 파일 로드: {df.shape}")

    df = build_derived_features(df)
    df = downcast(df)  # 새로 만든 파생변수들도 다운캐스트
    gc.collect()

    df.to_pickle(f"{OUT}/full_feature_table_derived.pkl")
    df.to_csv(f"{OUT}/full_feature_table_derived.csv", index=False, encoding=ENC)
    log(f"저장 완료: full_feature_table_derived.pkl / .csv ({df.shape})")

    del df
    gc.collect()
    log(f"전체 소요시간: {time.time()-t0:.1f}초")
