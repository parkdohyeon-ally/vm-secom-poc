"""Streamlit demo — wafer yield prediction from fab sensors.

Run:  streamlit run app.py     (after `python -m src.train_save`)
"""
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

from src.data import load_secom

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "vm_model.joblib"

st.set_page_config(page_title="Virtual Metrology · SECOM", layout="wide")
st.title("Virtual Metrology · 웨이퍼 불량 예측")
st.caption("fab 장비 센서 → 불량 확률 · 디스플레이 AI 자율실험실 → 반도체 VM PoC")

if not MODEL.exists():
    st.error("모델 아티팩트 없음 — 먼저 `python -m src.train_save` 실행하세요.")
    st.stop()


@st.cache_resource
def load():
    art = joblib.load(MODEL)
    X, y, ts = load_secom()
    return art["pipe"], X, y


pipe, X, y = load()

idx = st.slider("웨이퍼 선택 (index)", 0, len(X) - 1, 0)
row = X.iloc[[idx]]
prob = float(pipe.predict_proba(row)[0, 1])
actual = "FAIL" if y.iloc[idx] == 1 else "PASS"

c1, c2, c3 = st.columns(3)
c1.metric("예측 불량확률", f"{prob:.1%}")
c2.metric("예측 판정", "FAIL" if prob >= 0.5 else "PASS")
c3.metric("실제 라벨", actual)

clf = pipe.named_steps["clf"]
pre = pipe.named_steps["pre"]
names = X.columns[pre.named_steps["drop_missing"].keep_][pre.named_steps["var"].get_support()]
imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False).head(15)

st.subheader("주요 센서 (모델 중요도 Top-15)")
st.bar_chart(imp)
st.caption("센서 ID → 실제 공정 단계(증착·식각·CMP·포토) 매핑 시 공정 인사이트가 된다.")
