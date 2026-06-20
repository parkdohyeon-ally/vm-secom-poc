# Notebooks

순서대로 작성/실행:

1. **01_eda.ipynb** — 결측 맵, 클래스 불균형, 상수/근상수 센서, 상관 구조, 시간 분포
2. **02_preprocess.ipynb** — `src.features.build_preprocessor` 적용 전/후 비교, 차원 변화
3. **03_models.ipynb** — `run_baseline.py` 결과 재현 + 임계값(threshold) 튜닝, PR 커브
4. **04_explain.ipynb** — 최적 모델 SHAP → 상위 센서 → 공정 단계 해석

> 노트북은 `src/`의 함수를 import해서 쓰면 코드 중복이 없다.
> 예: `from src.data import load_secom`
