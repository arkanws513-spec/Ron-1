FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY core ./core
COPY model ./model
COPY api ./api
COPY app.py .
COPY index.html .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT}"]
