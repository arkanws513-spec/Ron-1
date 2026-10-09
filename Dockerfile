FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    RON_MODEL_ID=/data/models/Ron-1-Qwen3-1.7B \
    RON_MEMORY_PATH=/data/memory

WORKDIR /app

RUN useradd --create-home --uid 10001 ron \
    && mkdir -p /data \
    && chown ron:ron /data

COPY requirements-api.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.8.0" \
    && pip install --no-cache-dir -r requirements-api.txt

COPY --chown=ron:ron core ./core
COPY --chown=ron:ron model ./model
COPY --chown=ron:ron api ./api

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').getenv('PORT', '8000') + '/health', timeout=3).read()"

USER ron
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port \${PORT} --workers 1"]
