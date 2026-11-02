"""
=======================================================================================
STEP 3. 다중 테이블 병합·집계
=======================================================================================
목적:
    Home Credit 데이터는 7개 테이블이 SK_ID_CURR(고유ID)로 연결된 구조다.
    각 보조 테이블은 고객당 여러 행(과거 대출 여러 건, 매월 잔액 등)을 가지므로,
    "고객 한 명당 한 행"이 되도록 SK_ID_CURR 기준으로 집계(count/mean/max/sum)한 뒤
    메인 테이블(application_train/test)에 병합해야 분석·모델링이 가능하다.

병합 순서 (중요):
    1) bureau_balance(대출건별 월별 이력) -> SK_ID_BUREAU 단위로 먼저 집계
    2) 그 결과를 bureau(대출건 목록)에 합친 뒤 -> SK_ID_CURR 단위로 다시 집계
       (bureau_balance는 SK_ID_CURR가 없고 SK_ID_BUREAU만 있어서 2단계를 거쳐야 함)
    3) previous_application, POS_CASH_balance, credit_card_balance,
       installments_payments는 SK_ID_CURR가 이미 있어 바로 집계 가능
    4) 5개 집계 결과를 application_train/test에 순서대로 left join

사용 파일 (UP 폴더 안에 있어야 함):
    application_train_ko.csv, application_test_ko.csv,
    bureau_ko.csv, bureau_balance_ko.csv, previous_application_ko.csv,
    POS_CASH_balance_ko.csv, credit_card_balance_ko.csv, installments_payments_ko.csv
=======================================================================================
"""
import pandas as pd
import numpy as np
import time
import os
import gc

UP = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋"        # 한국어 번역 CSV들이 있는 폴더 (iCloud 동기화 문제로 Desktop 밖으로 이동)
OUT = "/Users/kang-youngju/homecredit_project/파이널 프로젝트 데이터셋/output"  # 결과물을 저장할 폴더
os.makedirs(OUT, exist_ok=True)

ENC = "utf-8-sig"  # 번역 파일들은 BOM 포함 utf-8로 저장되어 있음


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """int64/float64를 실제 값 범위에 맞는 가장 작은 타입으로 낮춰서 메모리를 줄인다.
    installments_payments.csv(723MB) 같은 큰 원본 파일을 읽을 때 특히 효과가 크다
    (보통 40~60% 메모리 절감). float64->float32는 소수점 정밀도가 약간 낮아지지만
    (유효숫자 약 7자리), 이 데이터의 금액·일수 단위에서는 실질적인 영향이 없다."""
    for c in df.select_dtypes(include="int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include="float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df


# ---------------------------------------------------------------------------
# 3-1. bureau_balance -> bureau -> SK_ID_CURR(고유ID) 단위 집계
# ---------------------------------------------------------------------------
def build_bureau_features():
    log("bureau_balance_ko.csv 집계 중...")
    bb = pd.read_csv(f"{UP}/bureau_balance_ko.csv", encoding=ENC,
                      dtype={"신용정보국대출ID": "int32", "기준월경과개월수": "int16", "연체상태": "category"})
    # 연체상태: 정상/종료/정보없음/연체1_30일/연체31_60일/연체61_90일/연체91_120일/연체120일초과또는대손
    bb["IS_DPD"] = bb["연체상태"].isin(
        ["연체1_30일", "연체31_60일", "연체61_90일", "연체91_120일", "연체120일초과또는대손"]
    ).astype("int8")
    bb_agg = bb.groupby("신용정보국대출ID").agg(
        BB_MONTHS_COUNT=("기준월경과개월수", "count"),   # 이력 개월수 (신용이력 길이)
        BB_MONTHS_MIN=("기준월경과개월수", "min"),       # 가장 오래된 기록 시점
        BB_DPD_SUM=("IS_DPD", "sum"),                    # 연체 발생 총 개월수
        BB_DPD_MEAN=("IS_DPD", "mean"),                  # 연체 발생 비율
    ).reset_index()
    del bb
    gc.collect()

    log("bureau_ko.csv 집계 중...")
    bureau = pd.read_csv(f"{UP}/bureau_ko.csv", encoding=ENC)
    bureau = downcast(bureau)
    bureau = bureau.merge(bb_agg, on="신용정보국대출ID", how="left")
    del bb_agg
    bureau["IS_ACTIVE"] = (bureau["대출상태"] == "활성").astype("int8")  # 대출상태: 활성/종료/매각/부실채권

    agg_spec = {
        "신용정보국대출ID": "count",                 # 과거 대출 건수
        "IS_ACTIVE": "sum",                          # 현재 활성 대출 건수
        "타기관대출신청경과일": ["min", "max", "mean"],
        "연체일수": ["max", "mean"],
        "총대출금액": ["sum", "mean", "max"],
        "대출잔액": ["sum", "mean"],
        "연체금액": ["sum", "mean"],
        "최대연체금액": "max",
        "대출연장횟수": "sum",
        "BB_DPD_SUM": "sum",
        "BB_DPD_MEAN": "mean",
    }
    bureau_agg = bureau.groupby("고유ID").agg(agg_spec)
    # 다중 인덱스 컬럼명을 "BUREAU_원본컬럼_집계함수" 형태로 평탄화
    bureau_agg.columns = ["BUREAU_" + "_".join(c).upper() for c in bureau_agg.columns]
    bureau_agg = bureau_agg.reset_index()
    del bureau
    gc.collect()
    return bureau_agg


# ---------------------------------------------------------------------------
# 3-2. previous_application -> 고유ID 단위 집계
# ---------------------------------------------------------------------------
def build_previous_application_features():
    log("previous_application_ko.csv 집계 중... (405MB, 시간이 좀 걸림)")
    prev = pd.read_csv(f"{UP}/previous_application_ko.csv", encoding=ENC)
    prev = downcast(prev)
    prev["IS_APPROVED"] = (prev["계약상태"] == "승인").astype("int8")
    prev["IS_REFUSED"] = (prev["계약상태"] == "거절").astype("int8")

    # 365243은 "값 없음"을 나타내는 플레이스홀더 이상값 -> 결측으로 처리
    for c in ["최초인출경과일", "최초상환예정경과일", "최초버전최종상환예정경과일",
              "최종상환예정경과일", "계약종료예정경과일"]:
        prev[c] = prev[c].replace(365243, np.nan)

    agg_spec = {
        "이전대출ID": "count",           # 과거 신청 건수
        "IS_APPROVED": "sum",           # 승인 건수
        "IS_REFUSED": "sum",            # 거절 건수
        "연간상환액": ["mean", "max"],
        "신청금액": ["mean", "max", "sum"],
        "최종승인금액": ["mean", "max", "sum"],
        "선수금": ["mean"],
        "심사결정경과일": ["min", "max", "mean"],
        "할부개월수": ["mean", "max"],
    }
    prev_agg = prev.groupby("고유ID").agg(agg_spec)
    prev_agg.columns = ["PREV_" + "_".join(c).upper() for c in prev_agg.columns]
    prev_agg = prev_agg.reset_index()
    del prev
    gc.collect()

    # 승인율/거절율 파생 (건수 대비 비율)
    prev_agg["PREV_APPROVAL_RATIO"] = prev_agg["PREV_IS_APPROVED_SUM"] / prev_agg["PREV_이전대출ID_COUNT"]
    prev_agg["PREV_REFUSAL_RATIO"] = prev_agg["PREV_IS_REFUSED_SUM"] / prev_agg["PREV_이전대출ID_COUNT"]
    return prev_agg


# ---------------------------------------------------------------------------
# 3-3. installments_payments -> 고유ID 단위 집계 (상환 이력)
# ---------------------------------------------------------------------------
def build_installments_features():
    log("installments_payments_ko.csv 집계 중... (723MB, 가장 큰 파일이라 가장 오래 걸림)")
    inst = pd.read_csv(f"{UP}/installments_payments_ko.csv", encoding=ENC)
    inst = downcast(inst)  # 723MB 원본은 downcast 없이 읽으면 메모리를 1.5~2GB까지 잡아먹을 수 있음
    inst["PAY_DELAY"] = inst["실제상환경과일"] - inst["할부예정상환경과일"]  # 양수=연체
    inst["PAY_DIFF"] = inst["할부예정금액"] - inst["실제상환금액"]           # 양수=미납
    inst["IS_LATE"] = (inst["PAY_DELAY"] > 0).astype("int8")

    agg_spec = {
        "이전대출ID": "count",
        "PAY_DELAY": ["mean", "max"],
        "PAY_DIFF": ["mean", "sum"],
        "IS_LATE": "mean",              # 연체 발생 비율
        "할부예정금액": ["mean", "sum"],
        "실제상환금액": ["mean", "sum"],
    }
    inst_agg = inst.groupby("고유ID").agg(agg_spec)
    inst_agg.columns = ["INST_" + "_".join(c).upper() for c in inst_agg.columns]
    inst_agg = inst_agg.reset_index()
    # 주의: count는 "총 할부납부 건수"이지 대출 건수가 아님 (대출 1건 = 할부가 여러 회차).
    # nunique로 "실제 몇 건의 대출에 대한 할부인지"를 별도로 추가
    nuniq = inst.groupby("고유ID")["이전대출ID"].nunique().rename("INST_이전대출ID_NUNIQUE")
    result = inst_agg.merge(nuniq, on="고유ID", how="left")
    del inst, inst_agg, nuniq
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# 3-4. POS_CASH_balance -> 고유ID 단위 집계 (매장/현금대출 월별 상태)
# ---------------------------------------------------------------------------
def build_pos_cash_features():
    log("POS_CASH_balance_ko.csv 집계 중... (392MB)")
    pos = pd.read_csv(f"{UP}/POS_CASH_balance_ko.csv", encoding=ENC)
    pos = downcast(pos)
    pos["IS_DPD"] = (pos["연체일수"] > 0).astype("int8")

    agg_spec = {
        "이전대출ID": "count",
        "연체일수": ["mean", "max"],
        "연체일수_유예제외": ["mean", "max"],
        "IS_DPD": "mean",
        "잔여할부개월수": "mean",
    }
    pos_agg = pos.groupby("고유ID").agg(agg_spec)
    pos_agg.columns = ["POS_" + "_".join(c).upper() for c in pos_agg.columns]
    pos_agg = pos_agg.reset_index()
    # 주의: count는 "총 월별 관측행 수"이지 대출 건수가 아님 (대출 1건이 여러 달 반복 관측됨).
    # nunique로 "실제 몇 건의 매장/현금대출인지"를 별도로 추가
    nuniq = pos.groupby("고유ID")["이전대출ID"].nunique().rename("POS_이전대출ID_NUNIQUE")
    result = pos_agg.merge(nuniq, on="고유ID", how="left")
    del pos, pos_agg, nuniq
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# 3-5. credit_card_balance -> 고유ID 단위 집계 (신용카드 월별 잔액)
# ---------------------------------------------------------------------------
def build_credit_card_features():
    log("credit_card_balance_ko.csv 집계 중... (424MB)")
    cc = pd.read_csv(f"{UP}/credit_card_balance_ko.csv", encoding=ENC)
    cc = downcast(cc)
    cc["UTILIZATION"] = cc["카드잔액"] / cc["신용한도액"].replace(0, np.nan)  # 카드이용률
    cc["IS_DPD"] = (cc["연체일수"] > 0).astype("int8")

    agg_spec = {
        "이전대출ID": "count",
        "카드잔액": ["mean", "max"],
        "UTILIZATION": ["mean", "max"],
        "연체일수": ["mean", "max"],
        "IS_DPD": "mean",
        "총인출금액": ["mean", "sum"],
        "총인출횟수": ["mean"],
    }
    cc_agg = cc.groupby("고유ID").agg(agg_spec)
    cc_agg.columns = ["CC_" + "_".join(c).upper() for c in cc_agg.columns]
    cc_agg = cc_agg.reset_index()
    # 주의: count는 "총 월별 관측행 수"이지 카드 개수가 아님 (카드 1개가 여러 달 반복 관측됨).
    # nunique로 "실제 몇 개의 카드를 보유했는지"를 별도로 추가
    nuniq = cc.groupby("고유ID")["이전대출ID"].nunique().rename("CC_이전대출ID_NUNIQUE")
    result = cc_agg.merge(nuniq, on="고유ID", how="left")
    del cc, cc_agg, nuniq
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# 3-6. 메인 테이블(application_train/test)에 5개 집계 결과를 모두 병합
# ---------------------------------------------------------------------------
def build_full_feature_table():
    app_train = pd.read_csv(f"{UP}/application_train_ko.csv", encoding=ENC)
    app_test = pd.read_csv(f"{UP}/application_test_ko.csv", encoding=ENC)

    # 성별="미상"(XNA) 4건 제거 -> application_train에만 있는 것으로 알려진
    # 데이터 자체의 이상값(성별이 기록되지 않은 극소수 케이스). 학습에 도움이 안 되고
    # 성별 관련 통계·시각화를 왜곡하므로 병합 전에 제거한다.
    before = len(app_train)
    app_train = app_train[app_train["성별"] != "미상"].reset_index(drop=True)
    log(f"성별='미상' {before - len(app_train)}건 제거 (application_train)")

    app_train["IS_TRAIN"], app_test["IS_TRAIN"] = 1, 0
    app_test["목표변수_상환어려움여부"] = np.nan  # test는 정답이 없으므로 결측으로 채워 구조를 맞춤
    app = pd.concat([app_train, app_test], axis=0, ignore_index=True, sort=False)
    app = downcast(app)
    del app_train, app_test
    gc.collect()
    log(f"메인 테이블 병합(train+test): {app.shape}")

    # 5개 보조 테이블은 한 번에 하나씩만 만들고 바로 병합·삭제한다.
    # (전부 먼저 만들어서 리스트에 쌓아두면 원본 CSV들이 순차적으로 메모리에서
    #  안 빠지고 남아있을 위험이 있어, 결과가 나오는 즉시 병합하고 지우는 방식으로 바꿈)
    for name, build_fn in [
        ("bureau", build_bureau_features),
        ("previous_application", build_previous_application_features),
        ("installments_payments", build_installments_features),
        ("POS_CASH_balance", build_pos_cash_features),
        ("credit_card_balance", build_credit_card_features),
    ]:
        agg_df = build_fn()
        app = app.merge(agg_df, on="고유ID", how="left")
        del agg_df
        gc.collect()
        log(f"{name} 병합 완료 -> {app.shape}")

    return app


if __name__ == "__main__":
    t0 = time.time()
    full_table = build_full_feature_table()
    log(f"최종 피처 테이블 완성: {full_table.shape} (소요시간 {time.time()-t0:.1f}초)")

    # 결과 저장 (다음 단계인 결측치·이상치 분석, EDA에서 이 파일을 그대로 불러와 이어서 사용)
    full_table.to_pickle(f"{OUT}/full_feature_table.pkl")   # 빠른 재로딩용 (downcast 덕에 용량도 더 작음)
    log(f"pkl 저장 완료: {OUT}/full_feature_table.pkl")
    full_table.to_csv(f"{OUT}/full_feature_table.csv", index=False, encoding="utf-8-sig")  # 확인용
    log(f"csv 저장 완료: {OUT}/full_feature_table.csv")
    del full_table
    gc.collect()
