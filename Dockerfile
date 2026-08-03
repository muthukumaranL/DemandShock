FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY config.yaml pyproject.toml ./
COPY .streamlit/ ./.streamlit/

ENV PYTHONPATH=/app/src

# The source datasets are NOT baked into the image (M5 terms). Mount them:
#   docker run -p 8501:8501 -v "$(pwd)/datasets:/app/datasets" demandshock
# Build the artifacts once inside the container before serving:
#   docker run -v "$(pwd)/datasets:/app/datasets" -v "$(pwd)/artifacts:/app/artifacts" \
#     demandshock python scripts/run_pipeline.py --mode development

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/Home.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
