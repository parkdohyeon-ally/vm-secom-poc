# Virtual Metrology PoC — 1-page Summary

## 문제
반도체 fab은 모든 웨이퍼를 계측할 수 없다. **장비 센서 데이터로 품질·계측값을 예측(Virtual Metrology, VM)** 하면 전수 측정 없이 수율을 관리하고 비용·시간을 줄인다.

## 데이터 & 방법
- **분류(수율)**: UCI SECOM — 1,567 웨이퍼 × 590 센서, 불량률 6.6%. 결측·고차원·불균형·시간drift 처리.
- **회귀(VM)**: 합성 증착 데이터(센서 → 막두께). 실제 XPS/박막 데이터로 교체 가능.
- 누수 없는 `Pipeline`(fold 내 fit), 불균형 정직 평가(PR-AUC·Recall@정밀도), PLS·RF·GBM·XGBoost.

## 결과
**수율 분류 (5-fold CV)** — no-skill PR-AUC = 0.066
| model | PR-AUC | ROC-AUC |
|---|---|---|
| RandomForest | **0.196** | **0.728** |
| PLS | 0.164 | 0.690 |
| Logistic | 0.149 | 0.663 |

**회귀 VM (5-fold CV)** — 막두께 예측
| model | RMSE (nm) | R² |
|---|---|---|
| **PLS** | **1.78** | **0.953** |
| GBM | 2.55 | 0.905 |
| RF | 3.53 | 0.819 |

→ 공선성 큰 공정 데이터에서 **PLS(VM 고전 기법)가 회귀 1위** — 도메인 정합 결과.

## 차별점 (왜 나인가)
- **계측·박막 도메인(XPS·TEM) + AI** 결합 — 순수 ML러도, 공정엔지니어도 아닌 가운데
- 누수·불균형·시간split 등 **현업 함정**을 설계 단계에서 처리
- 디스플레이 AI 자율실험실·디지털트윈 발주·설계 경험을 반도체 VM으로 직접 연결

## 이력서 한 줄
> **디스플레이 공정 데이터를 가상계측(VM) 문제로 모델링** — fab 센서 590변수 기반 수율 분류(PR-AUC 0.20, no-skill 대비 3×)와 막두께 회귀(PLS R² 0.95) PoC를 누수 없는 평가 파이프라인으로 구현. 박막 계측 도메인 + ML 융합.

깃허브: `github.com/parkdohyeon-ally/vm-secom-poc`
