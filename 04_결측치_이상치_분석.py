# -*- coding: utf-8 -*-
"""
STEP 4. 결측치·이상치 분석
- 목적: 데이터 품질 문제 파악 및 대치(imputation) 전략 수립
- 입력: 03_다중테이블_병합집계.py 가 만든 full_feature_table.pkl / .csv
- 출력: 결측치 리포트(csv), 이상치 리포트(csv), 대치 전략이 반영된 처리 후 테이블
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
    특히 csv로 다시 읽으면 pkl과 달리 타입 정보가 다 사라지고 float64/int64/object로
    복원되므로, master csv를 읽은 직후에는 반드시 한 번 실행해줘야 한다."""
    for c in df.select_dtypes(include="int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include="float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df


# ---------------------------------------------------------------------------
# 4-1. 결측치 현황 리포트
# ---------------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 결측 건수·비율·데이터타입을 정리한다. (신용조회건수 등 결측 자체가
    '이력 없음'이라는 정보가 되는 컬럼도 있으므로, 결측률만 보고 무조건 삭제하면
    안 되고 컬럼 성격을 함께 봐야 한다.)"""
    n = len(df)
    miss_cnt = df.isna().sum()
    rep = pd.DataFrame({
        "컬럼명": miss_cnt.index,
        "결측건수": miss_cnt.values,
        "결측비율(%)": (miss_cnt.values / n * 100).round(2),
        "데이터타입": [str(df[c].dtype) for c in miss_cnt.index],
    })
    rep = rep.sort_values("결측비율(%)", ascending=False).reset_index(drop=True)
    return rep


# ---------------------------------------------------------------------------
# 4-1-1. 근속후경과일(DAYS_EMPLOYED)의 365243 이상값 처리
#        - 365243일(=약 1000년)은 실제 값이 아니라 "무직/연금수급자"를 표시하기 위해
#          시스템이 넣어둔 특수 placeholder다. 실제 데이터로 확인해보니 전체의 약 18%
#          (64,648명)가 이 값을 갖고 있어, 그냥 두면 이상치 탐지·평균/중앙값 계산이
#          전부 심하게 왜곡된다. 통계적 이상치가 아니라 "코드값"이므로 이상치 제거
#          대상이 아니라, 별도 플래그로 분리해서 결측 처리하는 것이 맞는 방식이다.
# ---------------------------------------------------------------------------
def fix_days_employed_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    # 주의: 여기서는 df.copy()를 하지 않는다. 이 함수는 항상 impute_missing() 안에서만
    # 호출되는데, impute_missing()이 시작할 때 이미 한 번 복사를 해두므로 여기서 또
    # 복사하면 550MB짜리 데이터프레임을 불필요하게 두 번 복사하게 되어 메모리가
    # 두 배로 뛴다. 전달받은 df를 그대로 수정(in-place)한다.
    if "근속후경과일" in df.columns:
        anomaly_mask = df["근속후경과일"] == 365243
        df["무직_또는_연금수급여부"] = anomaly_mask.astype(int)
        df.loc[anomaly_mask, "근속후경과일"] = np.nan  # 이후 결측치 대치 로직에서 별도 처리
    return df


# ---------------------------------------------------------------------------
# 4-2. IQR 기반 이상치 탐지 (수치형 컬럼 전체 스캔)
# ---------------------------------------------------------------------------
def iqr_outlier_report(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Q1-1.5*IQR ~ Q3+1.5*IQR 범위를 벗어나는 값의 개수·비율을 컬럼별로 집계.
    상자그림(boxplot) 기준과 동일한 임계값을 수식으로 계산한 것."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_mask = (s < lo) | (s > hi)
        rows.append({
            "컬럼명": c, "Q1": q1, "Q3": q3, "IQR": iqr,
            "하한": lo, "상한": hi,
            "이상치건수": int(out_mask.sum()),
            "이상치비율(%)": round(out_mask.mean() * 100, 3),
        })
    return pd.DataFrame(rows).sort_values("이상치비율(%)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4-3. Isolation Forest 기반 다변량 이상치 탐지
#      (여러 컬럼을 동시에 봐야 드러나는 이상치 - 개별 컬럼은 정상범위여도
#       조합이 이상한 경우, 예: 소득은 낮은데 대출금액/연간상환액이 아주 큰 경우)
# ---------------------------------------------------------------------------
def isolation_forest_outliers(df: pd.DataFrame, feature_cols, contamination=0.01, random_state=42):
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer

    # 주의: out = df.copy()로 전체(수백 MB) 데이터프레임을 복사하지 않는다.
    # 실제로 필요한 건 고유ID + 이상치 판정 결과 2개 컬럼뿐이므로, 작은 결과표만 새로 만든다.
    X = df[feature_cols].values  # feature_cols만 뽑은 뷰(작은 배열), 복사 아님
    X = SimpleImputer(strategy="median").fit_transform(X)  # IF는 결측 허용 안 하므로 임시 중앙값 대치
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    pred = model.fit_predict(X)  # -1=이상치, 1=정상
    score = model.decision_function(X)  # 낮을수록 더 이상함
    result = pd.DataFrame({
        "고유ID": df["고유ID"].values,
        "IF_이상치여부": (pred == -1).astype(int),
        "IF_이상치점수": score,
    })
    return result


# ---------------------------------------------------------------------------
# 4-4. 결측치 대치 전략
# ---------------------------------------------------------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # (0) 근속후경과일 365243 placeholder 먼저 분리 (아래 결측 대치 로직에 자연스럽게 포함되도록)
    df = fix_days_employed_anomaly(df)

    # (A) 신용조회 관련 컬럼 - 결측 자체가 "조회 이력 없음"이라는 의미이므로 0으로 대치
    credit_inquiry_cols = [c for c in df.columns if c.startswith("신용조회건수_")]
    df[credit_inquiry_cols] = df[credit_inquiry_cols].fillna(0)

    # (A-1) 근속후경과일 - 위에서 365243을 결측으로 바꿔놨으므로, 같은 소득유형/직업종류
    #       그룹의 중앙값으로 대치 (무직자는 대부분 소득유형 자체가 "연금"/"무직"이라
    #       그룹을 나누면 자연스럽게 비슷한 사람들끼리 채워진다)
    if "근속후경과일" in df.columns:
        df["근속후경과일"] = df.groupby(["소득유형"])["근속후경과일"].transform(lambda s: s.fillna(s.median()))
        df["근속후경과일"] = df["근속후경과일"].fillna(df["근속후경과일"].median())

    # (B) 보조테이블(bureau/previous/inst/pos/cc) 병합 후 생긴 결측
    #     -> 해당 이력 자체가 없다는 뜻이므로 COUNT/NUNIQUE/SUM 계열은 0, 그 외 통계량(MEAN/MAX/MIN)은 결측 유지
    #        (0으로 채우면 "연체 0일"처럼 실제로는 없는 값이 생겨 왜곡되므로, 이런 컬럼은 대신
    #         '이력 유무' 플래그를 별도로 만들어 모델이 구분하게 한다)
    for prefix in ["BUREAU_", "PREV_", "INST_", "POS_", "CC_"]:
        cnt_cols = [c for c in df.columns if c.startswith(prefix) and ("COUNT" in c or "NUNIQUE" in c or "SUM" in c)]
        df[cnt_cols] = df[cnt_cols].fillna(0)
        has_flag = f"{prefix}이력있음"
        base_cnt_col = next((c for c in df.columns if c.startswith(prefix) and "COUNT" in c), None)
        if base_cnt_col:
            df[has_flag] = (df[base_cnt_col] > 0).astype(int)

    # (C) 건물 관련 비율 정보(아파트면적비율_평균 등 47개) - 동일 주거형태/소득유형 그룹의 중앙값으로 대치
    #     (그룹 정보가 비슷한 사람들끼리는 사는 건물 특성도 비슷할 가능성이 높다는 가정)
    building_cols = [c for c in df.columns if c.endswith(("_평균", "_최빈값", "_중앙값"))] + \
                     [c for c in ("전체면적비율",) if c in df.columns]
    for c in building_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df.groupby(["주거형태", "소득유형"])[c].transform(lambda s: s.fillna(s.median()))
            df[c] = df[c].fillna(df[c].median())  # 그룹으로도 못 채운 나머지는 전체 중앙값

    # (D) 외부신용점수1/2/3 - 대출상품종류별 중앙값으로 대치 (상품군마다 심사 기준이 다름)
    for c in ["외부신용점수1", "외부신용점수2", "외부신용점수3"]:
        if c in df.columns:
            df[c] = df.groupby("대출상품종류")[c].transform(lambda s: s.fillna(s.median()))
            df[c] = df[c].fillna(df[c].median())

    # (E) 직업종류 등 범주형 - 최빈값 대치
    for c in ["직업종류", "근무기관업종유형"]:
        if c in df.columns:
            mode_val = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_val.iloc[0] if len(mode_val) else "정보없음")

    return df


# ---------------------------------------------------------------------------
# 4-5. KNNImputer 기반 정밀 대치 (상관관계 높은 수치형 컬럼끼리 서로 참고)
# ---------------------------------------------------------------------------
def knn_impute(df: pd.DataFrame, cols, n_neighbors=5):
    from sklearn.impute import KNNImputer
    # 주의: df 전체를 복사하지 않고 필요한 cols만 복사해서 대치한 뒤 돌려준다.
    # 호출하는 쪽에서 df[cols] = knn_impute(df, cols) 형태로 다시 대입해서 쓰면 된다.
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols, index=df.index)


if __name__ == "__main__":
    t0 = time.time()

    # 03번은 더 이상 쓰지 않고, full_feature_table_imputed.csv를 마스터 파일로 바로 읽는다.
    # csv는 pkl과 달리 저장된 타입 정보가 없어 float64/int64/object로 복원되므로
    # downcast()를 반드시 바로 실행해서 메모리를 다시 줄여준다.
    df = pd.read_csv(f"{OUT}/full_feature_table_imputed.csv", encoding=ENC)
    df = downcast(df)
    log(f"마스터 파일 로드: {df.shape}")

    miss_rep = missing_value_report(df)
    miss_rep.to_csv(f"{OUT}/결측치_리포트.csv", index=False, encoding=ENC)
    log(f"결측치 리포트 저장 완료. 결측 있는 컬럼 수: {(miss_rep['결측건수'] > 0).sum()} / {len(miss_rep)}")
    del miss_rep

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rep = iqr_outlier_report(df, numeric_cols)
    outlier_rep.to_csv(f"{OUT}/이상치_IQR_리포트.csv", index=False, encoding=ENC)
    log(f"이상치 리포트 저장 완료. 이상치 존재 컬럼 수: {(outlier_rep['이상치건수'] > 0).sum()} / {len(outlier_rep)}")
    del outlier_rep

    df_imputed = impute_missing(df)   # 내부에서 df.copy()를 한 번만 하므로 이 시점에 두 벌이 됨
    del df                            # -> 원본은 더 이상 필요 없으니 바로 해제
    gc.collect()
    df_imputed.to_pickle(f"{OUT}/full_feature_table_imputed.pkl")
    log(f"결측치 대치 완료: {df_imputed.shape}")

    # Isolation Forest는 핵심 수치형 컬럼 일부만 사용(연산량 고려)
    key_cols = ["총소득", "대출금액", "연간상환액", "구매물품가격",
                "외부신용점수1", "외부신용점수2", "외부신용점수3",
                "출생후경과일", "근속후경과일"]
    key_cols = [c for c in key_cols if c in df_imputed.columns]
    if_result = isolation_forest_outliers(df_imputed, key_cols, contamination=0.01)  # 이제 작은 결과표만 반환됨
    log(f"Isolation Forest 이상치 탐지 완료. 이상치로 분류된 건수: {if_result['IF_이상치여부'].sum()}")
    if_result.to_csv(f"{OUT}/이상치_IsolationForest_리포트.csv", index=False, encoding=ENC)
    del df_imputed, if_result
    gc.collect()

    log(f"STEP 4 전체 소요시간: {time.time()-t0:.1f}초")
