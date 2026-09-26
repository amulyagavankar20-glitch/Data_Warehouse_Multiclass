# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /install

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

RUN pip install --upgrade pip

# Install Poetry
RUN pip install poetry==1.8.2

# Copy project files
COPY pyproject.toml ./

# Install Python dependencies (CPU-only torch)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null; \
    pip install \
    pandas numpy scikit-learn joblib pydantic pydantic-settings \
    mlflow fastapi "uvicorn[standard]" prometheus-client \
    matplotlib seaborn psycopg2-binary dvc prefect

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY src/ ./src/
COPY flows/ ./flows/
COPY run_pipeline.py .

# Create required directories
RUN mkdir -p data/raw data/processed mlruns models tests

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
