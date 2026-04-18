"""Parse a model's raw output into a list of labeled steps + final answer."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

STEP_RE = re.compile(r"(?mi)^\s*Step\s*(\d+)\s*[:.)\-]\s*(.+?)\s*$")
ANSWER_RE = re.compile(r"(?mi)^\s*Answer\s*[:\-]\s*(.+?)\s*$")
# Fallback: last number in the text, if the model didn't follow format.
TRAILING_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*\.?\s*$")


def parse_cot(text: str) -> Tuple[List[str], Optional[str]]:
    """Return (steps, answer).

    ``steps`` are the full lines including their ``Step N:`` prefix.
    ``answer`` is a short string (None if parsing fails).
    """
    if text is None:
        return [], None
    steps = [m.group(0).strip() for m in STEP_RE.finditer(text)]
    m = ANSWER_RE.search(text)
    answer = m.group(1).strip() if m else None

    if not steps:
        # Fallback: split on blank lines, drop any "Answer:" line.
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        steps = [c for c in chunks if not ANSWER_RE.match(c)]

    if answer is None:
        tm = TRAILING_NUMBER_RE.search(text.strip())
        if tm:
            answer = tm.group(1)

    return steps, answer


def steps_to_text(steps: List[str]) -> str:
    return "\n".join(steps)


def is_well_formed(steps: List[str], answer: Optional[str]) -> bool:
    return len(steps) >= 2 and answer is not None and len(answer) < 100
