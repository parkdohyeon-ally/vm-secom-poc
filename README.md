# Virtual Metrology PoC — Fab Sensor → Yield/Quality Prediction

반도체 fab 장비 센서 데이터로 웨이퍼 품질(양/불량)을 예측하는 **Virtual Metrology / FDC(Fault Detection & Classification)** 계열 PoC.
전수 계측이 불가능한 fab에서 "안 재고도 품질을 추정"하는 문제를 다룬다.

> 배경: 디스플레이 AI 자율실험실·디지털트윈 플랫폼(KDIA/EDIRAK)에서 공정 데이터 예측·연동 체계를 설계한 경험을
> 반도체 공정(증착·식각·CMP·포토)의 가상계측 문제로 확장한 개인 연구.

## 문제 정의
- 입력: 590개 장비 센서 변수 (공정 trace 요약)
- 출력: 웨이퍼 pass/fail (수율)
- 난점: **고차원 · 결측 · 클래스 불균형(~6.6% 불량) · 중복 센서 · 시간drift** — 실제 fab 데이터의 전형
- 의미: 불량을 조기에 잡으면 수율↑·재작업 비용↓

## 데이터
[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) — 1,567 웨이퍼 × 590 센서 + 타임스탬프. VM/FDC 벤치마크 표준.

## 빠른 시작
```bash
python -m venv .venv && source .venv/Scripts/activate   # win: .venv\Scripts\activate
pip install -r requirements.txt
python data/get_data.py        # SECOM 다운로드 → data/raw/
python run_baseline.py         # 전처리 + 모델 CV 비교 → reports/metrics.csv
```

## 구조
```
vm-secom-poc/
├─ data/get_data.py      # UCI 다운로드 (raw는 git 제외)
├─ src/
│  ├─ data.py            # 로드·라벨·타임스탬프
│  ├─ features.py        # 전처리(결측drop·대체·분산·스케일) — 누수 없는 Pipeline
│  ├─ models.py          # logreg / PLS / RF / XGBoost / LightGBM
│  └─ evaluate.py        # PR-AUC·ROC-AUC·Recall@정밀도, 누수 없는 CV
├─ run_baseline.py       # 엔드투엔드 실행
└─ notebooks/            # 01_eda · 02_preprocess · 03_models · 04_explain(SHAP)
```

## 방법론에서 신경 쓴 것 (성숙도 포인트)
- **데이터 누수 차단**: 결측 대체·스케일·피처선택을 전부 `Pipeline`에 넣어 **CV fold 안 train에서만 fit**
- **불균형 정직하게 평가**: Accuracy 금지. **PR-AUC / ROC-AUC / Recall@Precision** 사용
- **불균형 처리**: class weight · `scale_pos_weight` · 임계값 튜닝
- **시간drift**: fab 분포 변화 → 시간 기반 split 옵션 제공 (랜덤 split의 낙관 편향 회피)
- **설명가능성**: SHAP로 상위 센서 → "어느 공정 단계가 불량을 주도하는가" 공정 인사이트

## 결과 (실행 후 채움)
| model | PR-AUC | ROC-AUC | Recall@P=0.5 |
|---|---|---|---|
| logreg | – | – | – |
| pls | – | – | – |
| rf | – | – | – |
| xgb | – | – | – |

## 확장
- 비지도 이상탐지(IsolationForest·AutoEncoder)로 라벨 없는 FDC
- **회귀 VM**: 연속 계측값(막두께·CD) 예측 → RMSE·R² (정통 Virtual Metrology)
- Streamlit 데모: 센서 입력 → 예측 + SHAP

## 라이선스
MIT
