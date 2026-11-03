.PHONY: help install lint format test train repro serve drift mlflow-ui up down clean

help:
	@echo "make install    - uv sync (runtime + dev deps) and pre-commit hooks"
	@echo "make lint       - ruff, black --check, mypy"
	@echo "make format     - ruff --fix and black"
	@echo "make test       - pytest with 80% coverage gate"
	@echo "make train      - run the Prefect training flow"
	@echo "make repro      - dvc repro (retrain only if data/code changed)"
	@echo "make serve      - FastAPI on :8000 with autoreload"
	@echo "make drift BATCH=new.csv - PSI drift check against the trained bundle"
	@echo "make mlflow-ui  - MLflow UI on :5000 over ./mlflow.db"
	@echo "make up / down  - docker compose stack"

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests flows run_pipeline.py
	uv run black --check src tests flows run_pipeline.py
	uv run mypy src flows

format:
	uv run ruff check --fix src tests flows run_pipeline.py
	uv run black src tests flows run_pipeline.py

test:
	uv run pytest

train:
	uv run python run_pipeline.py

repro:
	uv run dvc repro

serve:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

drift:
	uv run python -m src.monitoring $(BATCH)

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

up:
	docker compose up -d --build

down:
	docker compose down

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
