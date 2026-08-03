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

# Neither the source datasets (M5 terms) nor the built artifacts are baked into the
# image. Build the artifacts once, mounting datasets plus the outputs:
#   docker run -v "$(pwd)/datasets:/app/datasets" -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/artifacts:/app/artifacts" -v "$(pwd)/models:/app/models" \
#     demandshock python scripts/run_pipeline.py --mode development
# Then serve, mounting the processed tables AND the artifacts the app reads:
#   docker run -p 8501:8501 -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/artifacts:/app/artifacts" demandshock
# Without both mounts the app starts and shows its "artifacts not built" panel.

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/Home.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
