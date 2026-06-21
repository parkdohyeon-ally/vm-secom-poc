# 배포 — HuggingFace Spaces (Docker SDK)

자가 부트스트랩(`src/bootstrap.py`)으로 빌드 시 데이터 다운로드 + 모델 학습까지 자동.

## 방법 A — 웹 UI (가장 쉬움)
1. https://huggingface.co/new-space
2. **SDK = Docker**, 이름 `vm-secom-demo`, Public
3. 생성된 Space에 이 파일들을 업로드/푸시:
   - `Dockerfile`, `requirements-app.txt`, `app.py`, `serve.py`, `src/`
   - `deploy/SPACE_README.md` → Space 루트에 **`README.md`**로 (YAML 헤더 필수)
4. 빌드 자동 시작 → 몇 분 후 데모 URL 활성화

## 방법 B — git push
```bash
# HF 토큰 로그인 (1회)
pip install huggingface_hub
huggingface-cli login

# Space repo 클론 후 파일 복사 푸시
git clone https://huggingface.co/spaces/<user>/vm-secom-demo
cp Dockerfile requirements-app.txt app.py serve.py vm-secom-demo/
cp -r src vm-secom-demo/
cp deploy/SPACE_README.md vm-secom-demo/README.md
cd vm-secom-demo && git add -A && git commit -m "deploy VM demo" && git push
```

## 로컬 Docker 테스트
```bash
docker build -t vm-demo .
docker run -p 7860:7860 vm-demo
# http://localhost:7860
```

## 주의
- Space **README.md의 YAML 헤더**(`sdk: docker`, `app_port: 7860`)가 없으면 빌드 안 됨
- 첫 빌드는 데이터 다운로드+학습 포함이라 수 분 소요
- 배포는 **네 HuggingFace 계정 인증** 필요 (토큰/로그인)
