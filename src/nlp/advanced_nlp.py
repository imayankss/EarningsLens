"""
src/nlp/advanced_nlp.py

Day 14 — Advanced NLP Features
Earnings Call Sentiment Analyzer

Features implemented:
    1. Keyword extraction       — top-N content words per transcript
    2. Uncertainty scoring      — hedging / epistemic uncertainty signal
    3. Topic frequency          — 7 financial topic buckets
    4. Bigram analysis          — top-N two-word collocations
    5. Speaker/section sentiment — CEO vs Analyst grouping (if available)

Design constraints:
    * No spaCy, no NLTK downloads — pure stdlib + pandas
    * All inputs optional; missing files produce warnings, not crashes
    * n=1 dataset handled gracefully throughout
    * Deterministic outputs (sorted, fixed seeds)
    * Atomic parquet writes (temp-rename pattern)
    * logging.getLogger(__name__) — no print in production paths
"""

from __future__ import annotations

import logging
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]

_TRANSCRIPTS_PATH = _ROOT / "data/interim/transcripts/transcripts_cleaned.parquet"
_CHUNKS_PATH      = _ROOT / "data/interim/chunks/chunks.parquet"
_FINBERT_PATH     = _ROOT / "data/processed/sentiment/finbert_chunk_scores.parquet"
_LM_PATH          = _ROOT / "data/processed/sentiment/lm_scores.parquet"
_MASTER_PATH      = _ROOT / "data/processed/master_dataset.parquet"

_NLP_OUT_DIR    = _ROOT / "data/processed/nlp"
_TABLES_OUT_DIR = _ROOT / "reports/tables"

_FEATURES_PARQUET      = _NLP_OUT_DIR / "advanced_nlp_features.parquet"
_FEATURES_CSV          = _NLP_OUT_DIR / "advanced_nlp_features.csv"
_KEYWORD_CSV           = _TABLES_OUT_DIR / "keyword_summary.csv"
_TOPIC_CSV             = _TABLES_OUT_DIR / "topic_frequency.csv"
_BIGRAM_CSV            = _TABLES_OUT_DIR / "bigram_summary.csv"
_UNCERTAINTY_CSV       = _TABLES_OUT_DIR / "uncertainty_summary.csv"
_SPEAKER_SENTIMENT_CSV = _TABLES_OUT_DIR / "speaker_sentiment_summary.csv"

# ---------------------------------------------------------------------------
# NLP vocabularies  (no downloads required)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "not", "no", "so", "yet", "both", "either",
    "neither", "nor", "that", "this", "these", "those", "than", "then",
    "when", "where", "which", "who", "whom", "whose", "what", "how",
    "why", "all", "any", "each", "every", "few", "more", "most", "other",
    "some", "such", "up", "out", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "against", "over",
    "under", "again", "further", "once", "here", "there", "just", "also",
    "very", "too", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "his", "her", "it", "its", "they", "their", "them", "us",
    "re", "ve", "ll", "d", "s", "t", "said", "get", "got", "go",
    "going", "come", "came", "make", "made", "take", "took", "think",
    "know", "want", "see", "look", "like", "well", "now", "one", "two",
    "three", "first", "second", "third", "new", "good", "great", "next",
    "last", "much", "many", "way", "can", "us", "per",
})

_UNCERTAINTY_WORDS: frozenset[str] = frozenset({
    "may", "might", "could", "possibly", "uncertain", "uncertainty",
    "risk", "risks", "volatile", "volatility", "depends", "estimate",
    "estimated", "estimates", "expected", "expects", "outlook",
    "guidance", "approximately", "potential", "potentially", "likely",
    "unlikely", "assume", "assumption", "assumes", "projected",
    "projection", "roughly", "subject", "contingent", "pending",
    "unclear", "unpredictable", "variable", "fluctuate", "fluctuation",
    "challenging", "headwind", "headwinds", "tailwind", "tailwinds",
    "cautious", "concern", "concerns",
})

_TOPIC_LEXICON: Dict[str, List[str]] = {
    "revenue_growth": [
        "revenue", "revenues", "growth", "grew", "grow", "sales",
        "topline", "acceleration", "accelerating", "demand",
    ],
    "margin_profitability": [
        "margin", "margins", "profitability", "profit", "profits",
        "gross", "operating", "ebitda", "earnings", "income",
    ],
    "guidance_outlook": [
        "guidance", "outlook", "forecast", "expect", "expects",
        "expected", "anticipate", "anticipates", "project", "projects",
        "projected", "quarter",
    ],
    "risk_uncertainty": [
        "risk", "risks", "uncertainty", "uncertain", "challenge",
        "challenges", "headwind", "headwinds", "volatile", "volatility",
        "concern", "concerns", "caution",
    ],
    "cost_expense": [
        "cost", "costs", "expense", "expenses", "spending", "investment",
        "opex", "capex", "efficiency", "savings", "reduction",
    ],
    "product_customer": [
        "product", "products", "customer", "customers", "service",
        "services", "demand", "adoption", "platform", "solution",
    ],
    "cash_debt_capital": [
        "cash", "debt", "capital", "balance", "sheet", "liquidity",
        "buyback", "dividend", "repurchase", "flow",
    ],
}

_TOP_N_KEYWORDS = 50
_TOP_N_BIGRAMS  = 30

# Pre-build topic term sets for O(1) lookup
_TOPIC_TERM_SETS: Dict[str, frozenset[str]] = {
    t: frozenset(words) for t, words in _TOPIC_LEXICON.items()
}

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class _TranscriptTokens:
    """Token-level representation of a single transcript."""
    transcript_id:  str
    ticker:         str
    tokens:         List[str]   # all tokens (lowercase alpha)
    content_tokens: List[str]   # stopwords removed


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_parquet_safe(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Return DataFrame or None; never raises."""
    if not path.exists():
        warnings.warn(
            f"[AdvancedNLP] Input not found, skipping '{label}': {path}",
            stacklevel=2,
        )
        return None
    try:
        df = pd.read_parquet(path)
        logger.info("Loaded %-30s : %d rows from %s", label, len(df), path.name)
        return df
    except Exception as exc:
        warnings.warn(
            f"[AdvancedNLP] Cannot read '{label}' ({path.name}): {exc}",
            stacklevel=2,
        )
        return None


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically via temp-rename."""
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase alphabetic word tokenisation — no external deps."""
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def _remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOPWORDS]


def _bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


# ---------------------------------------------------------------------------
# Column resolution utilities
# ---------------------------------------------------------------------------

def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_text_col(df: pd.DataFrame) -> Optional[str]:
    return _first_col(df, ["transcript_text_clean", "transcript_text", "text", "chunk_text"])


def _resolve_ticker_col(df: pd.DataFrame) -> Optional[str]:
    return _first_col(df, ["ticker", "company", "company_name"])


def _resolve_id_col(df: pd.DataFrame) -> Optional[str]:
    return _first_col(df, ["transcript_id", "id"])


# ---------------------------------------------------------------------------
# Token-record builder
# ---------------------------------------------------------------------------

def _build_token_records(
    df: pd.DataFrame,
    text_col: str,
    id_col: str,
    ticker_col: Optional[str],
) -> List[_TranscriptTokens]:
    """
    Aggregate all text rows per transcript_id, tokenise, strip stopwords.
    Returns one _TranscriptTokens per unique transcript.
    """
    records: List[_TranscriptTokens] = []
    for tid, grp in df.groupby(id_col, sort=True):
        combined = " ".join(
            str(v) for v in grp[text_col].dropna() if str(v).strip()
        )
        tokens  = _tokenize(combined)
        content = _remove_stopwords(tokens)

        ticker = ""
        if ticker_col and ticker_col in grp.columns:
            vals = grp[ticker_col].dropna()
            ticker = str(vals.iloc[0]) if len(vals) else ""

        records.append(
            _TranscriptTokens(
                transcript_id=str(tid),
                ticker=ticker,
                tokens=tokens,
                content_tokens=content,
            )
        )
    return records


# ---------------------------------------------------------------------------
# 1. Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(
    records: List[_TranscriptTokens],
    top_n: int = _TOP_N_KEYWORDS,
) -> pd.DataFrame:
    """
    Count content-word frequencies per transcript.

    Returns
    -------
    DataFrame with columns:
        transcript_id, ticker, keyword, count, frequency
    """
    rows: List[dict] = []
    for rec in records:
        if not rec.content_tokens:
            warnings.warn(
                f"[Keywords] No content tokens for {rec.transcript_id!r}; skipping.",
                stacklevel=2,
            )
            continue
        total   = len(rec.content_tokens)
        counter = Counter(rec.content_tokens)
        for kw, cnt in counter.most_common(top_n):
            rows.append({
                "transcript_id": rec.transcript_id,
                "ticker":        rec.ticker,
                "keyword":       kw,
                "count":         cnt,
                "frequency":     round(cnt / total, 6),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Uncertainty scoring
# ---------------------------------------------------------------------------

def compute_uncertainty(records: List[_TranscriptTokens]) -> pd.DataFrame:
    """
    Count hedging / epistemic uncertainty words per transcript.

    Returns
    -------
    DataFrame with columns:
        transcript_id, ticker, total_tokens, uncertainty_count, uncertainty_ratio
    """
    rows: List[dict] = []
    for rec in records:
        total     = len(rec.tokens)
        unc_count = sum(1 for t in rec.tokens if t in _UNCERTAINTY_WORDS)
        rows.append({
            "transcript_id":    rec.transcript_id,
            "ticker":           rec.ticker,
            "total_tokens":     total,
            "uncertainty_count": unc_count,
            "uncertainty_ratio": round(unc_count / total, 6) if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Topic frequency
# ---------------------------------------------------------------------------

def compute_topic_frequency(records: List[_TranscriptTokens]) -> pd.DataFrame:
    """
    Count hits per financial topic bucket per transcript.

    Returns
    -------
    DataFrame with columns:
        transcript_id, ticker, topic, count, ratio
    """
    rows: List[dict] = []
    for rec in records:
        total = len(rec.tokens) or 1
        for topic, term_set in _TOPIC_TERM_SETS.items():
            count = sum(1 for t in rec.tokens if t in term_set)
            rows.append({
                "transcript_id": rec.transcript_id,
                "ticker":        rec.ticker,
                "topic":         topic,
                "count":         count,
                "ratio":         round(count / total, 6),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Bigram analysis
# ---------------------------------------------------------------------------

def extract_bigrams(
    records: List[_TranscriptTokens],
    top_n: int = _TOP_N_BIGRAMS,
) -> pd.DataFrame:
    """
    Extract top-N two-word collocations (over content tokens) per transcript.

    Returns
    -------
    DataFrame with columns:
        transcript_id, ticker, bigram, count
    """
    rows: List[dict] = []
    for rec in records:
        if len(rec.content_tokens) < 2:
            warnings.warn(
                f"[Bigrams] Fewer than 2 content tokens for {rec.transcript_id!r}; skipping.",
                stacklevel=2,
            )
            continue
        counter = Counter(f"{a} {b}" for a, b in _bigrams(rec.content_tokens))
        for bigram, cnt in counter.most_common(top_n):
            rows.append({
                "transcript_id": rec.transcript_id,
                "ticker":        rec.ticker,
                "bigram":        bigram,
                "count":         cnt,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Speaker / section sentiment
# ---------------------------------------------------------------------------

def compute_speaker_sentiment(
    finbert_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Group FinBERT chunk-level sentiment by speaker role or section type.

    Returns
    -------
    (summary_df, issues)
        summary_df : DataFrame with avg sentiment per group, or empty
        issues     : list of human-readable warning strings
    """
    issues: List[str] = []

    if finbert_df is None or finbert_df.empty:
        issues.append(
            "FinBERT chunk scores not available — skipping speaker sentiment."
        )
        return pd.DataFrame(), issues

    group_col = _first_col(finbert_df, ["dominant_speaker", "speaker", "section_type"])
    if group_col is None:
        issues.append(
            "No speaker/section column found in FinBERT scores "
            "(looked for: dominant_speaker, speaker, section_type) — "
            "skipping speaker sentiment."
        )
        return pd.DataFrame(), issues

    score_col = _first_col(finbert_df, ["sentiment_score", "score"])
    if score_col is None:
        issues.append(
            "No sentiment score column found in FinBERT scores — "
            "skipping speaker sentiment."
        )
        return pd.DataFrame(), issues

    label_col = _first_col(finbert_df, ["predicted_label", "sentiment_label", "label"])
    id_col    = "transcript_id" if "transcript_id" in finbert_df.columns else None
    group_keys = [c for c in [id_col, group_col] if c is not None]

    rows: List[dict] = []
    for keys, grp in finbert_df.groupby(group_keys, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict = {k: v for k, v in zip(group_keys, keys)}
        row["chunk_count"]        = len(grp)
        row["avg_sentiment_score"] = round(float(grp[score_col].mean()), 6)
        row["std_sentiment_score"] = (
            round(float(grp[score_col].std()), 6) if len(grp) > 1 else 0.0
        )
        if label_col:
            lab = grp[label_col].astype(str).str.lower()
            row["positive_count"] = int((lab == "positive").sum())
            row["negative_count"] = int((lab == "negative").sum())
            row["neutral_count"]  = int((lab == "neutral").sum())
        rows.append(row)

    return pd.DataFrame(rows), issues


# ---------------------------------------------------------------------------
# Features summary (wide per-transcript)
# ---------------------------------------------------------------------------

def build_features_summary(
    uncertainty_df: pd.DataFrame,
    topic_df:       pd.DataFrame,
    keyword_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine per-transcript stats into a single wide DataFrame.

    Columns: transcript_id, ticker, total_tokens, uncertainty_count,
             uncertainty_ratio, topic_<name>…, top_5_keywords
    """
    if uncertainty_df.empty:
        return pd.DataFrame()

    base = uncertainty_df.copy()

    # Pivot topic counts wide
    if not topic_df.empty:
        topic_wide = topic_df.pivot_table(
            index="transcript_id",
            columns="topic",
            values="count",
            aggfunc="sum",
        ).reset_index()
        topic_wide.columns = [
            f"topic_{c}" if c != "transcript_id" else c
            for c in topic_wide.columns
        ]
        base = base.merge(topic_wide, on="transcript_id", how="left")

    # Top-5 keywords as comma-separated string
    if not keyword_df.empty:
        top_kw_rows: List[dict] = []
        for tid, grp in keyword_df.groupby("transcript_id", sort=True):
            top5 = ",".join(
                grp.nlargest(5, "count")["keyword"].tolist()
            )
            top_kw_rows.append({"transcript_id": tid, "top_5_keywords": top5})
        top_kw_df = pd.DataFrame(top_kw_rows)
        base = base.merge(top_kw_df, on="transcript_id", how="left")

    return base


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class AdvancedNLPPipeline:
    """
    Day 14 master pipeline.

    All input paths are optional.  Missing files emit warnings and the
    pipeline continues, producing partial (possibly empty) outputs.
    Handles n=1 single-transcript datasets gracefully.
    """

    def __init__(
        self,
        transcripts_path:  Path = _TRANSCRIPTS_PATH,
        chunks_path:       Path = _CHUNKS_PATH,
        finbert_path:      Path = _FINBERT_PATH,
        lm_path:           Path = _LM_PATH,
        master_path:       Path = _MASTER_PATH,
        nlp_out_dir:       Path = _NLP_OUT_DIR,
        tables_out_dir:    Path = _TABLES_OUT_DIR,
    ) -> None:
        self.transcripts_path = transcripts_path
        self.chunks_path      = chunks_path
        self.finbert_path     = finbert_path
        self.lm_path          = lm_path
        self.master_path      = master_path
        self.nlp_out_dir      = nlp_out_dir
        self.tables_out_dir   = tables_out_dir

    # ------------------------------------------------------------------
    def _load_inputs(self) -> Dict[str, Optional[pd.DataFrame]]:
        return {
            "transcripts": _load_parquet_safe(self.transcripts_path, "transcripts_cleaned"),
            "chunks":      _load_parquet_safe(self.chunks_path, "chunks"),
            "finbert":     _load_parquet_safe(self.finbert_path, "finbert_chunk_scores"),
            "lm":          _load_parquet_safe(self.lm_path, "lm_scores"),
            "master":      _load_parquet_safe(self.master_path, "master_dataset"),
        }

    # ------------------------------------------------------------------
    def _best_text_source(
        self, inputs: Dict[str, Optional[pd.DataFrame]]
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Optional[str]]:
        """
        Return (df, text_col, id_col, ticker_col) for the first usable source.
        Priority: transcripts → chunks → master.
        """
        for key in ("transcripts", "chunks", "master"):
            df = inputs.get(key)
            if df is None or df.empty:
                continue
            text_col   = _resolve_text_col(df)
            id_col     = _resolve_id_col(df)
            ticker_col = _resolve_ticker_col(df)
            if text_col and id_col:
                logger.info("Using '%s' as text source (col=%s)", key, text_col)
                return df, text_col, id_col, ticker_col
        return None, None, None, None

    # ------------------------------------------------------------------
    def _write_csv(self, df: pd.DataFrame, path: Path, label: str) -> None:
        if df is not None and not df.empty:
            df.to_csv(path, index=False)
            logger.info("Wrote %-35s : %d rows → %s", label, len(df), path.name)
        else:
            warnings.warn(
                f"[AdvancedNLP] '{label}' is empty; writing empty CSV placeholder.",
                stacklevel=2,
            )
            pd.DataFrame().to_csv(path, index=False)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the full Day-14 NLP pipeline.

        Returns
        -------
        dict mapping output name → DataFrame
        Empty dict if no usable text source is found.
        """
        _ensure_dirs(self.nlp_out_dir, self.tables_out_dir)
        inputs = self._load_inputs()

        df, text_col, id_col, ticker_col = self._best_text_source(inputs)
        if df is None:
            warnings.warn(
                "[AdvancedNLP] No usable text source found. "
                "Supply at least one of: transcripts_cleaned.parquet, "
                "chunks.parquet, or master_dataset.parquet.",
                stacklevel=2,
            )
            return {}

        records = _build_token_records(
            df, text_col=text_col, id_col=id_col, ticker_col=ticker_col
        )
        if not records:
            warnings.warn(
                "[AdvancedNLP] Token records are empty — check input data.",
                stacklevel=2,
            )
            return {}

        logger.info("Processing %d transcript(s)…", len(records))

        # ---- feature extraction ----------------------------------------
        keyword_df     = extract_keywords(records)
        uncertainty_df = compute_uncertainty(records)
        topic_df       = compute_topic_frequency(records)
        bigram_df      = extract_bigrams(records)
        speaker_df, speaker_issues = compute_speaker_sentiment(inputs.get("finbert"))

        for msg in speaker_issues:
            warnings.warn(f"[AdvancedNLP] {msg}", stacklevel=2)

        features_df = build_features_summary(uncertainty_df, topic_df, keyword_df)

        # ---- export --------------------------------------------------------
        if not features_df.empty:
            _atomic_parquet(features_df, self.nlp_out_dir / "advanced_nlp_features.parquet")
            features_df.to_csv(self.nlp_out_dir / "advanced_nlp_features.csv", index=False)
            logger.info(
                "Wrote advanced_nlp_features: %d rows", len(features_df)
            )

        self._write_csv(keyword_df,     self.tables_out_dir / "keyword_summary.csv",           "keyword_summary")
        self._write_csv(topic_df,       self.tables_out_dir / "topic_frequency.csv",           "topic_frequency")
        self._write_csv(bigram_df,      self.tables_out_dir / "bigram_summary.csv",            "bigram_summary")
        self._write_csv(uncertainty_df, self.tables_out_dir / "uncertainty_summary.csv",       "uncertainty_summary")
        self._write_csv(speaker_df,     self.tables_out_dir / "speaker_sentiment_summary.csv", "speaker_sentiment_summary")

        return {
            "features":                 features_df,
            "keyword_summary":          keyword_df,
            "topic_frequency":          topic_df,
            "bigram_summary":           bigram_df,
            "uncertainty_summary":      uncertainty_df,
            "speaker_sentiment_summary": speaker_df,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _configure_logging()
    logger.info("=== Day 14 — Advanced NLP Features ===")

    pipeline = AdvancedNLPPipeline()
    outputs  = pipeline.run()

    if not outputs:
        print(
            "\n[WARNING] No outputs produced.\n"
            "Ensure at least one input parquet file exists:\n"
            f"  {_TRANSCRIPTS_PATH}\n"
            f"  {_CHUNKS_PATH}\n"
            f"  {_MASTER_PATH}"
        )
        return

    print("\n=== Day 14 Advanced NLP — Output Summary ===")
    for name, df in outputs.items():
        rows = len(df) if (df is not None and not df.empty) else 0
        print(f"  {name:<40s}: {rows:>5d} rows")

    print("\nWritten files:")
    file_list = [
        _FEATURES_PARQUET, _FEATURES_CSV,
        _KEYWORD_CSV, _TOPIC_CSV, _BIGRAM_CSV,
        _UNCERTAINTY_CSV, _SPEAKER_SENTIMENT_CSV,
    ]
    for path in file_list:
        mark = "✓" if path.exists() else "✗"
        try:
            rel = path.relative_to(_ROOT)
        except ValueError:
            rel = path
        print(f"  {mark} {rel}")


if __name__ == "__main__":
    main()
