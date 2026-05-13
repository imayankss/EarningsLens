
from src.preprocessing.chunking import TranscriptChunker

sample_text = """
Good afternoon everyone and welcome to the earnings call.

Revenue increased 15% year-over-year.

We are seeing strong demand across enterprise customers.

Operator

We will now begin the question-and-answer session.
"""

chunker = TranscriptChunker()

chunks = chunker.chunk_transcript(
    sample_text,
    transcript_id="TEST_001"
)

print(f"\nTotal chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")

    if isinstance(chunk, dict):
        print("FIELDS:", list(chunk.keys()))

    print(chunk)
    print()
