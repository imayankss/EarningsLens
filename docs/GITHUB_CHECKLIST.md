# GitHub Checklist

## What to Commit

- Source code under `src/`
- `app.py`
- Tests under `tests/`
- Documentation under `docs/`
- Pipeline scripts under `scripts/`
- Required generated portfolio artifacts:
  - `data/processed/master_dataset.csv`
  - `data/processed/master_dataset.parquet`
  - `data/processed/nlp/`
  - `reports/tables/*.csv`
  - `reports/figures/*.png`
  - `models/predictive_model_metadata.json`

## What Not to Commit

- `.env`
- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.ruff_cache/`
- logs except `.gitkeep`
- raw private credentials or secrets
- `.streamlit/secrets.toml`

## Required Artifacts

Confirm these exist before publishing:

```bash
python3 scripts/repo_health_check.py
```

Important outputs:

- `data/processed/master_dataset.csv`
- `data/processed/master_dataset.parquet`
- `reports/tables/statistical_summary.csv`
- `reports/tables/correlation_analysis.csv`
- `reports/tables/regression_results.csv`
- `reports/tables/prediction_model_summary.csv`
- `reports/figures/*.png`

## Sanity Commands

```bash
python3 -m py_compile app.py
python3 -m pytest -q
python3 scripts/repo_health_check.py
```

## Dashboard Launch

```bash
streamlit run app.py
```

## README Screenshot Checklist

- Add dashboard overview screenshot:
  - `reports/screenshots/dashboard_overview.png`
- Add sentiment page screenshot if desired:
  - `reports/screenshots/dashboard_sentiment.png`
- Add prediction page screenshot if desired:
  - `reports/screenshots/dashboard_prediction.png`

Do not fake screenshots. Add them only after capturing the real dashboard.

## Final Health Command

```bash
python3 scripts/repo_health_check.py
```

Expected current warning:

```text
master_dataset has only 1 row; inference and ML validation remain limited.
```
