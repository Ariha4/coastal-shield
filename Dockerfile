# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt prometheus-client

COPY . .

# 8501 = Streamlit UI, 8000 = Prometheus metrics endpoint
EXPOSE 8501
EXPOSE 8000

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
