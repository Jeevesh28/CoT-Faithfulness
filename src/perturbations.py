"""Four perturbation functions that take a list of CoT steps and return
a perturbed version + character spans of the changed tokens (for attention
alignment later)."""
from __future__ import annotations

import random
import re
from typing import List, Tuple

Spans = List[Tuple[int, int]]

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _joined_char_span(steps: List[str], step_idx: int, local_start: int, local_end: int) -> Tuple[int, int]:
    """Map (step_idx, local offset) to a char span in ``"\n".join(steps)``."""
    offset = sum(len(s) + 1 for s in steps[:step_idx])  # +1 per newline
    return (offset + local_start, offset + local_end)


# ----------------------------------------------------------------------------
# (i) Step Corruption
# ----------------------------------------------------------------------------

def step_corruption(steps: List[str], seed: int = 0) -> Tuple[List[str], Spans]:
    """Replace one intermediate number with a plausibly-wrong number.

    * Targets a non-last step (we don't overwrite the final-answer step).
    * Keeps integer vs. float type to stay realistic.
    """
    rng = random.Random(seed)
    if len(steps) < 2:
        return list(steps), []

    candidates = []
    for i, s in enumerate(steps[:-1]):
        for m in _NUM_RE.finditer(s):
            candidates.append((i, m))
    if not candidates:
        return list(steps), []

    i, m = rng.choice(candidates)
    orig = m.group()
    if "." in orig:
        wrong = f"{float(orig) + rng.uniform(2.0, 9.0) * rng.choice([-1, 1]):.2f}"
    else:
        delta = rng.choice([-13, -7, -3, 3, 7, 13])
        wrong = str(int(orig) + delta)
    new_step = steps[i][:m.start()] + wrong + steps[i][m.end():]
    new_steps = steps[:i] + [new_step] + steps[i + 1:]
    span = _joined_char_span(new_steps, i, m.start(), m.start() + len(wrong))
    return new_steps, [span]


# ----------------------------------------------------------------------------
# (ii) Wrong-but-Consistent CoT
# ----------------------------------------------------------------------------

def wrong_but_consistent(steps: List[str], donor_steps: List[str]) -> Tuple[List[str], Spans]:
    """Replace the CoT entirely with a valid CoT for a different problem.

    ``donor_steps`` is a well-formed CoT produced by the same model on a
    different problem. The whole CoT is marked as ``perturbed`` for attention
    alignment.
    """
    full = "\n".join(donor_steps)
    return list(donor_steps), [(0, len(full))]


# ----------------------------------------------------------------------------
# (iii) Early Termination
# ----------------------------------------------------------------------------

def early_termination(steps: List[str], keep_frac: float = 0.5) -> Tuple[List[str], Spans]:
    """Truncate to the first ``keep_frac`` of the steps (at least one step)."""
    k = max(1, int(round(len(steps) * keep_frac)))
    return steps[:k], []


# ----------------------------------------------------------------------------
# (iv) Step Shuffling
# ----------------------------------------------------------------------------

def step_shuffling(steps: List[str], seed: int = 0) -> Tuple[List[str], Spans]:
    """Randomly permute the step order."""
    if len(steps) <= 1:
        return list(steps), []
    rng = random.Random(seed)
    idx = list(range(len(steps)))
    # Guarantee at least one position changes.
    for _ in range(10):
        rng.shuffle(idx)
        if idx != list(range(len(steps))):
            break
    return [steps[i] for i in idx], []


PERTURBATIONS = {
    "step_corruption": step_corruption,
    "wrong_but_consistent": wrong_but_consistent,
    "early_termination": early_termination,
    "step_shuffling": step_shuffling,
}
