# Rwanda Weather MLOps API: FastAPI + TensorFlow (CPU) + web dashboard
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Non-root user with id 1000 (required by Hugging Face Spaces, good practice everywhere)
RUN useradd -m -u 1000 app
WORKDIR /app

# Dependencies first, so code changes don't reinstall TensorFlow
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY data/train/ data/train/
COPY data/test/ data/test/
RUN mkdir -p data/uploads && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
