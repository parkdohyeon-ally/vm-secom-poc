# Virtual Metrology demo — HuggingFace Spaces (Docker SDK) / generic container
FROM python:3.11-slim

WORKDIR /app

# minimal runtime deps (no jupyter/shap/xgboost) for a small, fast image
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# app code
COPY src ./src
COPY app.py serve.py ./

# pre-bake: download SECOM + train model at build time (offline at runtime)
RUN python -m src.bootstrap

# HuggingFace Spaces expects 7860
EXPOSE 7860
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0"]
