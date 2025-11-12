FROM python:3.11-slim

WORKDIR /app

# System updates (optional) for compatibility
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install retriever deps
RUN pip install -U pip && \
    pip install fastapi uvicorn psycopg[binary] sentence-transformers

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "scripts.retriever_api:app", "--host", "0.0.0.0", "--port", "8000"]
