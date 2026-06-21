# Roadmap — 대규모 데이터 확장 + 제품화

## 데이터 확장 (단계적)
| 단계 | 데이터 | 규모/성격 | 추가 가치 | 접근 |
|---|---|---|---|---|
| 0 (완료) | UCI SECOM | 1.5k×590, 수율 분류 | 방법론 baseline | UCI 공개 |
| **1** | **Bosch Production Line** | **1.18M×968, 극불균형** | 대규모·실데이터 신뢰, out-of-core 처리 | Kaggle 대회 |
| 2 | PHM 2016 **CMP** | 실제 CMP 제거율 | **정통 회귀 VM**(합성 대체) | PHM Society |
| 3 | **WM-811K** 웨이퍼맵 | 811k 장, 결함 9종 | 결함패턴 CNN = 제품성 | Kaggle |

### 1단계 — Bosch (지금 코드로 바로)
- `src/datasets/bosch.py` — 샘플/청크 로더 완비
- 기존 `build_preprocessor` + tree 모델 그대로 사용
- 포인트: **out-of-core**(청크 집계)·극불균형(0.58%)·**MCC**(Bosch 공식 지표) 추가
- 다운로드: `kaggle competitions download -c bosch-production-line-performance -f train_numeric.csv.zip -p data/raw`

### 2단계 — PHM2016 CMP (회귀) — 코드 완료, 데이터 대기
- `src/datasets/phm_cmp.py` — 센서 트레이스 → wafer 단위 집계 로더
- `notebooks/07_phm_cmp_vm.ipynb` — 회귀 파이프라인 재사용, RMSE·R²·parity
- `src/regression.py`의 합성 데이터를 **실제 CMP 제거율**로 교체 (PHM 데이터 받으면 그대로 실행)

### 3단계 — WM-811K (이미지) — 코드 완료, 데이터 대기
- `src/datasets/wm811k.py` — LSWMD.pkl 로더, 가변맵 → 64×64 리사이즈
- `src/cnn.py` — PyTorch CNN, 클래스 가중 손실, macro-F1
- `notebooks/08_wafermap_cnn.ipynb` — 학습·평가·혼동행렬
- `requirements-cnn.txt` (torch 등 별도, 코어 경량 유지)
- 사업성: 웨이퍼맵 결함패턴 자동 분류 → 근본원인 추적

## 제품화 패키징 (진행)
- [x] **FastAPI 서빙** `serve.py` (`/predict`, `/health`)
- [x] **ROI 계산기** `src/roi.py` (recall→$ 절감)
- [x] **Model Card** `MODEL_CARD.md`
- [x] Streamlit 대시보드 `app.py`
- [ ] 배포: HuggingFace Spaces / Render (무료 호스팅)
- [ ] CI: GitHub Actions (lint·test·smoke)
- [ ] Dockerfile

## 비즈니스 프레이밍
- 가치 = **수율↑·재작업↓·계측비↓**. ROI 계산기로 정량화.
- 타깃 고객: fab 수율·공정기술팀, 장비사 APC SW, 산업AI(가우스랩스류)
