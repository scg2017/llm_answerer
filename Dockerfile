FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ENV LISTEN_PORT=5000 \
    DB_PATH=/data/answer_cache.db

RUN mkdir -p /data

EXPOSE 5000

CMD ["python", "llm_answerer.py"]
