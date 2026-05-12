#!/usr/bin/env bash
# =============================================================================
# Earnings Call Sentiment Analyzer — Day 1 Setup
# macOS Apple Silicon (M1/M2/M3) | Python 3.12
# Usage: chmod +x setup_day1.sh && ./setup_day1.sh
# =============================================================================
set -e

echo ""
echo "============================================================"
echo "  Earnings Call Sentiment Analyzer — Day 1 Setup"
echo "============================================================"
echo ""

# Check we are in the project root
if [ ! -f "pyproject.toml" ]; then
  echo "ERROR: Run this script from the project root directory."
  echo "  cd earnings-call-sentiment-analyzer && ./setup_day1.sh"
  exit 1
fi

# Check Python 3.12
if ! command -v python3.12 &>/dev/null; then
  echo "Python 3.12 not found. Install it first:"
  echo "  brew install python@3.12"
  exit 1
fi

echo ">>> Python version: $(python3.12 --version)"

# Create virtual environment
echo ""
echo ">>> Creating .venv with Python 3.12..."
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo ">>> Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ">>> Installing dependencies (5-10 min first run)..."
pip install torch torchvision torchaudio --quiet
pip install transformers datasets tokenizers accelerate sentencepiece --quiet
pip install pandas pyarrow duckdb polars numpy --quiet
pip install yfinance pandas-market-calendars --quiet
pip install nltk scikit-learn statsmodels scipy --quiet
pip install plotly streamlit matplotlib seaborn --quiet
pip install pyyaml python-dotenv loguru tqdm requests httpx joblib --quiet
pip install pytest pytest-cov pytest-mock black isort flake8 --quiet
pip install jupyter ipykernel nbformat --quiet

echo ""
echo ">>> Setup complete!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  pytest tests/ -v"
echo "  python scripts/run_pipeline.py --max-rows 50 --tickers AAPL MSFT --skip-lm"
echo "  streamlit run app/streamlit_app.py"
echo ""
echo "LM Dictionary (download separately):"
echo "  https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
echo "  Save to: data/dictionaries/loughran_mcdonald/LM_MasterDictionary.csv"
echo "============================================================"
