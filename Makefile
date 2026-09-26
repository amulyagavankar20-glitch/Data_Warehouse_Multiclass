.PHONY: install lint format test train serve mlflow-ui pipeline clean help

help:
	@echo "Available targets:"
	@echo "  make install      - Install dependencies via Poetry"
	@echo "  make lint         - Run ruff, black --check, and mypy"
	@echo "  make format       - Run ruff --fix and black"
	@echo "  make test         - Run pytest with coverage"
	@echo "  make train        - Run Prefect training flow"
	@echo "  make serve         - Start FastAPI server"
	@echo "  make mlflow-ui    - Start MLflow UI on :5000"
	@echo "  make pipeline     - Run DVC pipeline (dry-run)"
	@echo "  make clean        - Remove caches and artifacts"

install:
	poetry install

lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	black src/ tests/

test:
	pytest tests/ --cov=src/ --cov-report=term-missing --cov-fail-under=80

train:
	python run_pipeline.py

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

pipeline:
	dvc repro --dry

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov mlruns mlruns/ mlartifacts/
	rm -f mlflow.db
