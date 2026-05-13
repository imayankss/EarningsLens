import pandas as pd

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.preprocessing.chunking import ChunkingConfig, TranscriptChunker


sample_text = """
Operator

Welcome everyone to the quarterly earnings conference call.

John Smith - Chief Executive Officer

Thank you everyone for joining us today.
We delivered very strong quarterly performance across all segments.
Revenue increased significantly year over year driven by strong customer demand,
higher retention, improved operational efficiency, and expansion into new markets.

Our cloud division performed exceptionally well this quarter and margins improved
due to disciplined cost optimization initiatives. Enterprise adoption trends remain strong.

We also continued investing in artificial intelligence infrastructure and long-term
platform capabilities. Customer engagement metrics improved across all regions.

Jane Doe - Chief Financial Officer

Operating income increased substantially compared to the prior year.
Cash flow remained strong and balance sheet quality improved materially.

We reduced debt, expanded free cash flow generation,
and maintained strong liquidity throughout the quarter.

Looking ahead, we remain optimistic about demand trends,
customer expansion opportunities, and long-term profitability.

Operator

We will now begin the question-and-answer session.

Michael Lee - Goldman Sachs

Can you discuss guidance expectations for next quarter and margin outlook?

John Smith - Chief Executive Officer

We expect continued momentum next quarter with stable demand trends,
improving margins, and continued enterprise adoption across products.

We remain confident in long-term execution and growth opportunities.
"""

df = pd.DataFrame(
    [
        {
            "transcript_id": "AAPL_20240130",
            "ticker": "AAPL",
            "earnings_date": "2024-01-30",
            "transcript_text": sample_text,
        }
    ]
)


pipeline = PreprocessingPipeline()
pipeline.chunker = TranscriptChunker(
    ChunkingConfig(
        min_tokens=20,
        target_tokens=80,
        max_tokens=120,
        overlap_tokens=10,
        sentence_tokenizer="regex",
    )
)

result = pipeline.run(df)

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)

for key, value in result.summary.items():
    print(f"{key}: {value}")

print("\n" + "=" * 60)
print("TRANSCRIPTS DF")
print("=" * 60)
print(result.transcripts_df.head())

print("\n" + "=" * 60)
print("SEGMENTS DF")
print("=" * 60)
print(result.segments_df.head())

print("\n" + "=" * 60)
print("CHUNKS DF")
print("=" * 60)
print(result.chunks_df.head())

print("\n" + "=" * 60)
print("VALIDATION REPORT")
print("=" * 60)
print(result.validation_report.head())
