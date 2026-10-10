FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY core /app/core
RUN python -m pip install --no-cache-dir -r /app/core/requirements.txt
EXPOSE 8080
CMD ["python", "/app/core/server.py"]
