.PHONY: install preprocess train-phase1 train-phase2 evaluate test lint clean

install:
	pip install -e ".[dev]"

preprocess:
	python scripts/preprocess.py

train-phase1:
	python scripts/train_phase1.py

train-phase2:
	python scripts/train_phase2.py

generate:
	python scripts/generate.py

evaluate:
	python scripts/evaluate.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

