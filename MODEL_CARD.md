# Model Card — Virtual Metrology (Yield Classifier)

## 개요
fab 장비 센서 변수로 웨이퍼 **불량(fail) 확률**을 예측하는 분류 모델. 전수 계측 없이 수율을 추정하는 Virtual Metrology / FDC 용도.

## 의도된 사용
- 인라인 품질 스크리닝(고위험 웨이퍼 우선 계측·검사)
- 공정 모니터링·이상 알림 보조
- **비의도**: 안전·합격 최종 판정 단독 근거로 사용 금지 (사람 검토 보조용)

## 학습 데이터
- UCI **SECOM** — 1,567 웨이퍼 × 590 센서, 불량률 6.6% (2008, 반도체 fab)
- 확장: Bosch(118만), PHM2016 CMP(회귀) — 로드맵 참고

## 메트릭 (5-fold CV)
| | PR-AUC | ROC-AUC |
|---|---|---|
| RandomForest | 0.196 | 0.728 |
- no-skill PR-AUC = 0.066. 불균형 데이터라 **Accuracy 미사용**.

## 평가 방법
- 누수 차단: 전처리(대체·스케일·피처선택)를 fold 내 train에서만 fit
- 불균형: PR-AUC·Recall@정밀도, class weight / scale_pos_weight
- 운영 임계값은 비용(미검출 vs 오탐)에 따라 조정

## 한계·리스크
- SECOM은 소규모·구형 → 절대 성능보다 **방법론 신뢰성** 위주
- 센서 분포 drift 시 재학습 필요(시간 split로 점검)
- 도메인 매핑(센서→공정 단계) 없으면 인사이트 제한
- 데이터 편향: 단일 fab·기간

## 운영
- 재학습: `python -m src.train_save`
- 서빙: `uvicorn serve:app` (`POST /predict`)
- 모니터링: 입력 분포·예측 분포 drift, recall 추적

## 책임
작성: parkdohyeon-ally · 용도: 연구/포트폴리오 PoC
