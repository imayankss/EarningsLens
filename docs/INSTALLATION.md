# Installation

## Python Version

Python 3.12 is recommended. The project was developed with modern pandas, PyArrow, scikit-learn, Streamlit, and transformer/NLP dependencies.

## Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Install Dependencies

For normal usage:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## Loughran-McDonald Dictionary

The LM baseline expects the master dictionary at:

```text
data/dictionaries/loughran_mcdonald/LM_MasterDictionary.csv
```

If the file is missing, download it from the official Notre Dame Software Repository for Accounting and Finance and place it at that path.

## Launch Dashboard

```bash
streamlit run app.py
```

## Verify Repository Health

```bash
python3 scripts/repo_health_check.py
```

## Run Tests

```bash
python3 -m pytest -q
```

## Common Troubleshooting

### `ModuleNotFoundError`

Confirm you are running commands from the repository root and that dependencies are installed in the active virtual environment.

### Missing LM dictionary

Place `LM_MasterDictionary.csv` in `data/dictionaries/loughran_mcdonald/`.

### Streamlit command not found

Install dependencies and activate the virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Statistical or ML outputs are skipped

This is expected for the current one-event dataset. The statistical and prediction modules are guarded and will run meaningful tests/training only after enough events are available.
