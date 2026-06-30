FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv
COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt --system

COPY . .

RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "src.mlops_project.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
