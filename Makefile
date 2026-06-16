.PHONY: install test test-fast pipeline pipeline-fast lint clean

PYTHON ?= python3.12
VENV_PYTHON := .venv/bin/python
PIP := .venv/bin/pip

install:
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

test:
	$(VENV_PYTHON) -m pytest -q

test-fast:
	$(VENV_PYTHON) -m pytest -q tests/unit tests/test_event_study.py tests/test_finbert_pipeline.py

pipeline:
	$(VENV_PYTHON) scripts/run_pipeline.py

pipeline-fast:
	$(VENV_PYTHON) scripts/run_pipeline.py --max-rows 50 --tickers AAPL MSFT --skip-lm

lint:
	$(VENV_PYTHON) -m py_compile $$(find src tests scripts app -name '*.py' -print)

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache
