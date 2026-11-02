# -*- coding: utf-8 -*-
"""
STEP 4-대안. 결측치 처리 - "결측치_처리_유형별_정리" 가이드 버전 (v2)
04_결측치_이상치_분석.py의 impute_missing()과는 규칙이 조금씩 다른 대안 버전이다.
같은 데이터에 두 방식을 각각 적용해서 05_모델비교_LR_RF_XGB_LGBM.py로 AUC를 비교해보기 위한 것.

가이드 5유형과 처리 규칙 (PDF 원문 기준):
  ① 외부신용점수1/2/3      : 전체 median 대치 + "_MISSING" 플래그
  ② 건물·주거 통계 42개    : median 금지, HAS_BUILDING_INFO 플래그 + 0 대치
  ③ 이력집계 COUNT/SUM/NUNIQUE(BUREAU_/PREV_/POS_/INST_/CC_) : 0 대치 (이력없음=0건)
  ④ 이력집계 MEAN/MAX/RATIO(BUREAU_/PREV_/POS_/INST_/CC_)    : HAS_X 플래그 유지 + 0 대치
                                                                (04단계 v1과 다른 점: v1은 NaN 유지, v2는 0 대치)
  ⑵ 일반 소량 결측(차량연식/신용조회건수/사회관계망 연체지인수 등) : 관련 플래그 컬럼으로 판단 후 0/median
  마지막 : ①~⑤로 처리되지 않고 남은 결측 컬럼은 median으로 일괄 마무리
  주의   : XGBoost/LightGBM/CatBoost는 원본 결측 유지본(대치 전 master csv)도 함께 남겨서
           "결측을 직접 대치 vs 모델이 스스로 처리" 두 방식을 비교할 수 있게 한다.

입력: full_feature_table_imputed.csv (마스터 파일, 03번은 더 이상 사용하지 않음 - 04번과 동일 입력)
출력: full_feature_table_imputed_v2.pkl (04번의 full_feature_table_imputed.pkl과 별도 이름으로 저장되어
      05/06번에서 파일명만 바꿔주면 두 방식을 서로 비교할 수 있음)
"""
import time
import gc
import numpy as np
import pandas as pd

UP = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋"  # iCloud 동기화 문제로 Desktop 밖으로 이동
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"
ENC = "utf-8-sig"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """int64/float64를 실제 값 범위에 맞는 가장 작은 타입으로 낮춰서 메모리를 줄인다.
    csv로 읽으면 타입 정보가 사라져 float64/int64/object로 복원되므로 읽은 직후 실행."""
    for c in df.select_dtypes(include="int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include="float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df


# ---------------------------------------------------------------------------
# 근속후경과일 365243 placeholder는 유형 분류와 무관하게 먼저 처리 (04단계와 동일)
# 주의: 아래 5개 impute_typeN_* 함수는 전부 df.copy()를 하지 않는다.
#       impute_missing_by_type()이 맨 처음에 딱 한 번만 복사해두고, 그 뒤로는
#       같은 사본을 계속 이어받아 제자리(in-place)에서 수정한다.
#       전에는 함수마다 각자 df.copy()를 해서 550MB+짜리 데이터프레임을 5번
#       연달아 복사하고 있었는데(호출될 때마다 이전 사본+새 사본이 잠깐 동시에
#       존재해 순간 메모리 사용량이 두 배로 뛰었음), 이제 복사는 한 번으로 끝난다.
# ---------------------------------------------------------------------------
def fix_days_employed_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    if "근속후경과일" in df.columns:
        anomaly_mask = df["근속후경과일"] == 365243
        df["무직_또는_연금수급여부"] = anomaly_mask.astype(int)
        df.loc[anomaly_mask, "근속후경과일"] = np.nan
    return df


# ---------------------------------------------------------------------------
# 유형 ① 외부신용점수1/2/3 - median 대치 + _MISSING 플래그
#         (+ 여유가 있으면: 서로 상관관계를 이용한 회귀 대치도 옵션으로 제공)
# ---------------------------------------------------------------------------
def impute_type1_external_score(df: pd.DataFrame, use_regression=True) -> pd.DataFrame:
    cols = ["외부신용점수1", "외부신용점수2", "외부신용점수3"]
    cols = [c for c in cols if c in df.columns]

    for c in cols:
        df[f"{c}_MISSING"] = df[c].isna().astype(int)

    if use_regression and len(cols) >= 2:
        # 나머지 점수들로 결측 점수를 선형회귀 추정 (정규방정식, sklearn 불필요)
        for target in cols:
            others = [c for c in cols if c != target]
            known = df[[target] + others].dropna()
            if len(known) < 100:
                continue
            X = np.column_stack([np.ones(len(known)), known[others].values])
            y = known[target].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)

            need = df[target].isna() & df[others].notna().all(axis=1)
            if need.sum() > 0:
                X_pred = np.column_stack([np.ones(need.sum()), df.loc[need, others].values])
                # downcast()로 float32가 된 컬럼에 float64 예측값을 그대로 넣으면
                # pandas 최신 버전에서 dtype 경고(향후 에러 예정)가 나므로 명시적으로 맞춰줌
                df.loc[need, target] = (X_pred @ beta).astype(df[target].dtype)
                log(f"{target}: 회귀 대치 {need.sum()}건 (나머지 점수 {others} 이용)")

    # 회귀로도 못 채운 나머지(다른 점수들도 전부 결측인 극소수)는 median으로 마무리
    for c in cols:
        df[c] = df[c].fillna(df[c].median())

    return df


# ---------------------------------------------------------------------------
# 유형 ② 건물·주거 통계 - median 금지, HAS_BUILDING_INFO 플래그 + 0 대치
# ---------------------------------------------------------------------------
def impute_type2_building_info(df: pd.DataFrame) -> pd.DataFrame:
    building_cols = [c for c in df.columns if c.endswith(("_평균", "_최빈값", "_중앙값"))]
    building_cols += [c for c in ("전체면적비율",) if c in df.columns]
    if not building_cols:
        return df

    # 이 42개 컬럼은 "공동주택 정보 자체가 없는 사람"에게서 한꺼번에 결측되는 경향이 있어
    # 공통 플래그 하나로 충분한지 먼저 확인 후, 공통 플래그 + 개별 0 대치로 처리
    has_info = df[building_cols].notna().any(axis=1)
    df["HAS_BUILDING_INFO"] = has_info.astype(int)
    df[building_cols] = df[building_cols].fillna(0)
    log(f"건물·주거 통계 {len(building_cols)}개 컬럼: HAS_BUILDING_INFO 플래그 생성 + 0 대치 "
        f"(정보 있는 비율 {has_info.mean()*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# 유형 ③④ 이력집계 (BUREAU_/PREV_/POS_/INST_/CC_)
#   ③ COUNT/SUM/NUNIQUE -> 0 대치
#   ④ MEAN/MAX/RATIO/MIN 등 나머지 -> HAS_X 플래그 유지 + 0 대치 (04단계 v1과 다른 지점)
# ---------------------------------------------------------------------------
def impute_type3_4_history_agg(df: pd.DataFrame) -> pd.DataFrame:
    for prefix in ["BUREAU_", "PREV_", "INST_", "POS_", "CC_"]:
        prefix_cols = [c for c in df.columns if c.startswith(prefix)]
        if not prefix_cols:
            continue

        # HAS_X 플래그: 이 접두사 계열 컬럼 중 하나라도 값이 있으면 "이력 있음"
        has_flag = f"HAS_{prefix.rstrip('_')}"
        df[has_flag] = df[prefix_cols].notna().any(axis=1).astype(int)

        cnt_cols = [c for c in prefix_cols if ("COUNT" in c or "SUM" in c or "NUNIQUE" in c)]
        rest_cols = [c for c in prefix_cols if c not in cnt_cols]

        df[cnt_cols] = df[cnt_cols].fillna(0)   # 유형 ③
        df[rest_cols] = df[rest_cols].fillna(0)  # 유형 ④ (플래그로 진짜 0과 결측을 구분하므로 0 대치 허용)

    return df


# ---------------------------------------------------------------------------
# 유형 ⑤ 일반 소량 결측 - 관련 플래그 컬럼으로 판단 후 0/median
# ---------------------------------------------------------------------------
def impute_type5_misc(df: pd.DataFrame) -> pd.DataFrame:
    # 차량연식: 자가용소유여부가 "아니오"인 사람은 원래부터 해당사항 없음 -> 0 + 플래그
    #           자가용소유여부가 "예"인데 결측인 경우만 median으로 대치
    if "차량연식" in df.columns and "자가용소유여부" in df.columns:
        df["차량정보있음"] = df["차량연식"].notna().astype(int)
        no_car = df["자가용소유여부"].astype(str).isin(["아니오", "N", "no", "0"])
        df.loc[no_car & df["차량연식"].isna(), "차량연식"] = 0
        has_car_missing = (~no_car) & df["차량연식"].isna()
        if has_car_missing.any():
            df.loc[has_car_missing, "차량연식"] = df["차량연식"].median()

    # 신용조회건수_* : 결측=조회 이력 없음 -> 0
    credit_inquiry_cols = [c for c in df.columns if c.startswith("신용조회건수_")]
    df[credit_inquiry_cols] = df[credit_inquiry_cols].fillna(0)

    # 사회관계망 연체 관측/발생 지인수 : 결측 = 주변에 연체자가 없거나 조사 안 됨 -> 0
    social_cols = [c for c in df.columns if c.endswith("연체관측지인수") or c.endswith("연체발생지인수")]
    df[social_cols] = df[social_cols].fillna(0)

    # 신청동반자유형(범주형) : 최빈값 대치
    if "신청동반자유형" in df.columns:
        mode_val = df["신청동반자유형"].mode(dropna=True)
        df["신청동반자유형"] = df["신청동반자유형"].fillna(mode_val.iloc[0] if len(mode_val) else "정보없음")

    # 그 외 범주형(건물유지보수적립방식/건물외벽재질/건물유형/비상상태신고이력/직업종류/근무기관업종유형) : 최빈값
    for c in ["건물유지보수적립방식", "건물외벽재질", "건물유형", "비상상태신고이력",
              "직업종류", "근무기관업종유형"]:
        if c in df.columns and df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_val.iloc[0] if len(mode_val) else "정보없음")

    # 그 외 일반 소량 결측 수치형(가족구성원수/전화번호변경후경과일/구매물품가격/연간상환액 등) : median
    misc_numeric = ["가족구성원수", "전화번호변경후경과일", "구매물품가격", "연간상환액"]
    for c in misc_numeric:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    return df


# ---------------------------------------------------------------------------
# 전체 파이프라인: ① -> ⑤ 순서대로 적용 -> 마지막 남은 결측은 median 일괄 마무리
# ---------------------------------------------------------------------------
def impute_missing_by_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # 호출한 쪽의 원본은 건드리지 않도록 여기서 딱 한 번만 복사
    df = fix_days_employed_anomaly(df)
    df = impute_type1_external_score(df)
    df = impute_type2_building_info(df)
    df = impute_type3_4_history_agg(df)
    df = impute_type5_misc(df)

    # 근속후경과일은 365243 제거 후에도 남는 결측을 median으로 마무리 (①~⑤에 명시 안 된 항목)
    if "근속후경과일" in df.columns and df["근속후경과일"].isna().any():
        df["근속후경과일"] = df["근속후경과일"].fillna(df["근속후경과일"].median())

    # 마지막 안전장치: ①~⑤ 규칙에 안 걸린 채 남은 결측 컬럼(수치형은 median, 범주형은 최빈값)
    # 주의: 목표변수(TARGET)는 test 행이라 원래부터 결측인 것이 정상이므로 반드시 제외해야 함
    #       (여기서 채워버리면 예측해야 할 test 라벨을 0으로 조작하는 심각한 오류가 됨)
    remaining = [c for c in df.columns[df.isna().any()].tolist() if c != "목표변수_상환어려움여부"]
    if remaining:
        log(f"①~⑤ 규칙에 안 걸리고 남은 결측 컬럼 {len(remaining)}개 -> median/최빈값으로 마무리: {remaining}")
        for c in remaining:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                mode_val = df[c].mode(dropna=True)
                df[c] = df[c].fillna(mode_val.iloc[0] if len(mode_val) else "정보없음")

    return df


if __name__ == "__main__":
    t0 = time.time()

    # 03번은 더 이상 쓰지 않고, 04번과 동일하게 full_feature_table_imputed.csv를
    # 마스터 파일로 바로 읽는다. csv는 타입 정보가 없어 float64/int64/object로
    # 복원되므로 downcast()로 다시 줄여준다.
    df = pd.read_csv(f"{OUT}/full_feature_table_imputed.csv", encoding=ENC)
    df = downcast(df)
    log(f"마스터 파일 로드: {df.shape}")

    df_v2 = impute_missing_by_type(df)
    del df  # impute_missing_by_type 내부에서 이미 복사본을 만들어 처리했으므로 원본은 해제
    gc.collect()
    log(f"유형별 가이드(v2) 대치 완료: {df_v2.shape}")
    assert df_v2.drop(columns=["목표변수_상환어려움여부"]).isna().sum().sum() == 0, "목표변수 외에 결측이 남아있음"

    df_v2.to_pickle(f"{OUT}/full_feature_table_imputed_v2.pkl")
    log(f"저장 완료: {OUT}/full_feature_table_imputed_v2.pkl")
    del df_v2
    gc.collect()
    log(f"전체 소요시간: {time.time()-t0:.1f}초")
    log("이 파일을 05_모델비교_LR_RF_XGB_LGBM.py나 06_EDA_통계검정.py에서 "
        "full_feature_table_imputed.pkl 대신 full_feature_table_imputed_v2.pkl을 읽게 "
        "바꿔서 돌리면 04단계 방식과 v2 방식을 바로 비교할 수 있음")
