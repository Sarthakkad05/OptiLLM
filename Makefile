.PHONY: help dev test test-fast bench docker-up docker-down lint format train-router clean

PYTHON ?= python3
VENV ?= venv
BIN = $(VENV)/bin

help:
	@echo "OptiLLM — Developer & Operational Commands"
	@echo "=========================================="
	@echo "  make dev          - Start the local gateway dev server with hot reload"
	@echo "  make test         - Run full test suite with pytest"
	@echo "  make test-fast    - Run unit tests only without external service deps"
	@echo "  make bench        - Run automated latency, cache, and router benchmarks"
	@echo "  make docker-up    - Start full stack in Docker (OptiLLM, DB, Redis, Prom, Grafana)"
	@echo "  make docker-down  - Stop all running Docker containers"
	@echo "  make lint         - Run ruff linter and style checks"
	@echo "  make format       - Format code with black and ruff"
	@echo "  make train-router - Train / retrain the local ML router model"
	@echo "  make clean        - Remove Python cache files and temporary artifacts"

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

test-fast:
	pytest tests/unit/ -v

bench:
	python3 -m benchmarks.generate_report

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

lint:
	ruff check app/ tests/ optillm_client/
	black --check app/ tests/ optillm_client/

format:
	black app/ tests/ optillm_client/
	ruff check app/ tests/ optillm_client/ --fix

train-router:
	python3 scripts/train_router.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -f .coverage coverage.xml test-results.xml
