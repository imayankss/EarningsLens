import re

pattern = re.compile(
r"(?m)^(?:(?P<operator>operator\s*:?)|(?P<structured>[A-Z][A-Za-z\u2019'-.]+(?:\s+[A-Z][A-Za-z\u2019'-.]+){0,4}\s*[-\u2013\u2014]\s*[A-Za-z][A-Za-z\s,.&/()]{2,80}))\s*$",
re.MULTILINE | re.IGNORECASE,
)

sample = """
Operator

Welcome everyone.

John Smith - Chief Executive Officer

Great quarter.

David Lee - Goldman Sachs
"""

matches = list(pattern.finditer(sample))

print("MATCH COUNT =", len(matches))
print(matches)

