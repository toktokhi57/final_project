# 결과 파일 (results)

코드를 실행하지 않고도 핵심 수치와 그래프를 볼 수 있도록 대표 결과만 모았습니다.
원본 데이터와 고객 단위 예측 결과는 포함하지 않았고, 모두 집계된 요약표입니다.
핵심 그림 3장(SHAP 요약, 확률 보정 곡선, 위험 집단 카드)은 `../images/`에 있으며 아래 그림 목록에도 함께 실었습니다. 표 파일 이름을 누르면 바로 열립니다.

## tables

| 주제 | 파일 | 생성 스크립트 |
|---|---|---|
| EDA·전처리 | [`EDA_수치형변수_통계검정.csv`](tables/EDA_수치형변수_통계검정.csv), [`EDA_범주형변수_통계검정.csv`](tables/EDA_범주형변수_통계검정.csv) | `06` |
| | [`결측치_리포트.csv`](tables/결측치_리포트.csv) | `04` |
| 모델 비교 | [`모델비교_기준모델_Dummy_로지스틱.csv`](tables/모델비교_기준모델_Dummy_로지스틱.csv) | `10` |
| | [`모델비교_불균형처리_RandomForest.csv`](tables/모델비교_불균형처리_RandomForest.csv) | `11` |
| | [`모델비교_머신러닝12종.csv`](tables/모델비교_머신러닝12종.csv) | `12` |
| | [`CV_비교결과_5fold.csv`](tables/CV_비교결과_5fold.csv), [`하이퍼파라미터튜닝_결과.csv`](tables/하이퍼파라미터튜닝_결과.csv) | `13` |
| | [`파생변수_효과검증.csv`](tables/파생변수_효과검증.csv) | `32` |
| 해석 | [`SHAP_상위변수_TOP20.csv`](tables/SHAP_상위변수_TOP20.csv) | `16` |
| 임계값·비용비율 | [`임계값_최적화_결과.csv`](tables/임계값_최적화_결과.csv) | `14` |
| | [`임계값_비용비율_최종근거표.csv`](tables/임계값_비용비율_최종근거표.csv) (비용비율별 임계값 비교, 채택 임계값 0.665는 5:1 구간과 일치) | `14_2` (저장소 미포함) |
| | [`실제배율_계산결과.csv`](tables/실제배율_계산결과.csv) (이자계산 3.22:1) | `15` (저장소 미포함) |
| | [`LGD순이익률_배율_계산결과.csv`](tables/LGD순이익률_배율_계산결과.csv) (5.21:1), [`LGD순이익률_기반_임계값_결과.csv`](tables/LGD순이익률_기반_임계값_결과.csv) | `15_1` |
| 확률 보정·등급 | [`확률보정_비교_결과.csv`](tables/확률보정_비교_결과.csv) | `17` |
| | [`등급변환_방법1_인원균등분할.csv`](tables/등급변환_방법1_인원균등분할.csv), [`등급변환_방법2_고정확률구간.csv`](tables/등급변환_방법2_고정확률구간.csv), [`등급변환_방법3_기저율중심.csv`](tables/등급변환_방법3_기저율중심.csv) (방법3 채택) | `18` |
| | [`등급별_프로파일링_수치형.csv`](tables/등급별_프로파일링_수치형.csv) | `19` |
| | [`등급별_의사결정규칙.csv`](tables/등급별_의사결정규칙.csv) | `23` |
| 검증 | [`공정성_성별검증_결과.csv`](tables/공정성_성별검증_결과.csv) | `20` |
| | [`최종테스트셋_검증_결과.csv`](tables/최종테스트셋_검증_결과.csv), [`최종테스트셋_등급별검증.csv`](tables/최종테스트셋_등급별검증.csv) | `21` |
| 위험 집단 | [`위험집단_요약.txt`](tables/위험집단_요약.txt), [`위험집단_규칙.csv`](tables/위험집단_규칙.csv), [`위험집단_카드.csv`](tables/위험집단_카드.csv), [`위험집단_등급기준표.csv`](tables/위험집단_등급기준표.csv) | `31` |
| 세그먼트 (통합 파이프라인) | [`SEG_1_단변량_범주형.csv`](tables/SEG_1_단변량_범주형.csv), [`SEG_1_단변량_수치형.csv`](tables/SEG_1_단변량_수치형.csv) | `30` |
| | [`SEG_2_축선정.csv`](tables/SEG_2_축선정.csv) (SHAP + 단변량 교집합으로 분할 축 선정) | `30` |
| | [`SEG_3_규칙표.csv`](tables/SEG_3_규칙표.csv), [`SEG_3_깊이비교.csv`](tables/SEG_3_깊이비교.csv) (의사결정나무 규칙) | `30` |
| | [`SEG_4_군집특징.csv`](tables/SEG_4_군집특징.csv), [`SEG_4_군집프로파일.csv`](tables/SEG_4_군집프로파일.csv) (K-Means 보조 군집화) | `30` |
| | [`SEG_5_교차검증.csv`](tables/SEG_5_교차검증.csv), [`SEG_6_공정성점검.csv`](tables/SEG_6_공정성점검.csv) | `30` |
| | [`SEG_7_최종세그먼트.csv`](tables/SEG_7_최종세그먼트.csv) (S1\~S8 + 기대손익·대응전략) | `30` |

> 참고: `임계값_비용비율_최종근거표.csv`의 비고 열에는 4:1 행이 "최종 채택 대표값"으로 적혀 있습니다. 다만 실제로 채택한 임계값 0.665는 5:1·5.21:1 행(0.666)과 거의 일치하며, 4:1 행의 임계값은 0.696입니다.

## figures
그림을 바로 볼 수 있도록 펼쳐 두었습니다. 원본 파일은 `figures/` 폴더에 있습니다.

### EDA

**EDA 상관관계 히트맵** (`06`)

<img src="figures/EDA_상관관계_히트맵.png" alt="EDA 상관관계 히트맵" width="720">

**범주형 변수 부도율 바차트** (`06`)

<img src="figures/EDA_범주형_부도율_바차트.png" alt="범주형 변수 부도율 바차트" width="720">

### 모델링

**불균형 처리 AUC·Recall 비교** (`11`)

<img src="figures/불균형처리_AUC_Recall_비교.png" alt="불균형 처리 AUC·Recall 비교" width="720">

**머신러닝 12종 ROC (트리 계열)** (`12`)

<img src="figures/모델비교_머신러닝12종_ROC_트리계열.png" alt="머신러닝 12종 ROC (트리 계열)" width="720">

**머신러닝 12종 ROC (선형·거리 계열)** (`12`)

<img src="figures/모델비교_머신러닝12종_ROC_선형거리계열.png" alt="머신러닝 12종 ROC (선형·거리 계열)" width="720">

### 임계값·비용비율

**임계값 최적화 ROC** (`14`)

<img src="figures/임계값_최적화_ROC.png" alt="임계값 최적화 ROC" width="720">

**비용비율 최종 근거 그래프** (`14_2 (저장소 미포함)`)

<img src="figures/임계값_비용비율_최종근거_그래프.png" alt="비용비율 최종 근거 그래프" width="720">

### 해석

**SHAP summary plot** (`16`)

<img src="../images/shap_summary.png" alt="SHAP summary plot" width="720">

**SHAP waterfall (고위험)** (`16`)

<img src="figures/SHAP_waterfall_고위험.png" alt="SHAP waterfall (고위험)" width="720">

**SHAP waterfall (저위험)** (`16`)

<img src="figures/SHAP_waterfall_저위험.png" alt="SHAP waterfall (저위험)" width="720">

### 확률 보정·등급

**확률 보정 신뢰도 곡선** (`17`)

<img src="../images/calibration_curve.png" alt="확률 보정 신뢰도 곡선" width="720">

**등급 변환 3방법 비교** (`18`)

<img src="figures/등급변환_3방법_비교.png" alt="등급 변환 3방법 비교" width="720">

**등급별 프로파일링 박스플롯** (`19`)

<img src="figures/등급별_프로파일링_박스플롯.png" alt="등급별 프로파일링 박스플롯" width="720">

**등급별 기대손익 그래프** (`23`)

<img src="figures/등급별_기대손익_그래프.png" alt="등급별 기대손익 그래프" width="720">

### 검증

**성별 공정성 검증 비교** (`20`)

<img src="figures/공정성_성별검증_비교.png" alt="성별 공정성 검증 비교" width="720">

### 위험 집단

**위험 집단 카드** (`31`)

<img src="../images/risk_group_cards.png" alt="위험 집단 카드" width="720">

**위험 집단 그래프** (`31`)

<img src="figures/위험집단_그래프.png" alt="위험 집단 그래프" width="720">

**위험 집단 의사결정나무** (`31`)

<img src="figures/위험집단_나무.png" alt="위험 집단 의사결정나무" width="720">

### 세그먼트 (통합 파이프라인)

**세그먼트 그림1: 단변량** (`30`)

<img src="figures/SEG_그림1_단변량.png" alt="세그먼트 그림1: 단변량" width="720">

**세그먼트 그림2: 의사결정나무** (`30`)

<img src="figures/SEG_그림2_나무.png" alt="세그먼트 그림2: 의사결정나무" width="720">

**세그먼트 그림3: 군집** (`30`)

<img src="figures/SEG_그림3_군집.png" alt="세그먼트 그림3: 군집" width="720">

**세그먼트 그림4: 최종 요약** (`30`)

<img src="figures/SEG_그림4_최종요약.png" alt="세그먼트 그림4: 최종 요약" width="720">

세그먼트 고객별 배정 결과(`SEG_할당결과.pkl`)는 고객 단위 데이터라 포함하지 않았습니다.
