print("START")

from src.preprocessing.transcript_segmenter import TranscriptSegmenter

print("IMPORTED")

segmenter = TranscriptSegmenter()

print("INITIALIZED")

sample = """
Operator

Welcome everyone.

John Smith - Chief Executive Officer

Great quarter.

Operator

We will now begin the question-and-answer session.

David Lee - Goldman Sachs

Congrats on results.
"""

segments = segmenter.segment_transcript(
raw_text=sample,
transcript_id="TEST_001"
)

print("DONE")
print("SEGMENT COUNT:", len(segments))
print(segments)

