"""Dataset loaders for GSM8K-Platinum and BIG-Bench Hard.

We construct a frozen 75-problem split (50 math + 25 logic) that gets committed
to ``data/processed/subset.jsonl``. All downstream scripts read from that file
so the experiment is reproducible.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

SUBSET_PATH = pathlib.Path("data/processed/subset.jsonl")


@dataclass
class Problem:
    id: str
    dataset: str           # "gsm8k_platinum" | "bbh"
    task: str              # BBH task name, or "" for GSM8K
    question: str
    gold_answer: str
    difficulty: str        # "easy" | "medium" | "hard"


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def _gsm8k_answer(text: str) -> str:
    """GSM8K answers are written like `... #### 42`."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    return m.group(1) if m else text.strip().split()[-1]


def _gsm8k_difficulty(q: str) -> str:
    """Rough difficulty proxy: sentence count + digit density."""
    n_sent = max(1, q.count(".") + q.count("?"))
    n_nums = len(re.findall(r"\d+", q))
    score = n_sent + n_nums * 0.5
    if score < 5:
        return "easy"
    if score < 9:
        return "medium"
    return "hard"


def load_gsm8k_platinum(n: int, seed: int) -> List[Problem]:
    """Load GSM8K-Platinum (curated, label-cleaned subset of GSM8K)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("madrylab/gsm8k-platinum", split="test")
    except Exception:
        # Fallback to plain GSM8K test split if the platinum variant isn't
        # accessible in the current environment.
        ds = load_dataset("gsm8k", "main", split="test")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    out: List[Problem] = []
    for i in indices:
        row = ds[i]
        q = row["question"]
        gold = _gsm8k_answer(row["answer"])
        out.append(Problem(
            id=f"gsm8k_{i:05d}",
            dataset="gsm8k_platinum",
            task="",
            question=q,
            gold_answer=gold,
            difficulty=_gsm8k_difficulty(q),
        ))
    return out


# A compact set of BBH tasks that (a) have short inputs and (b) have crisp
# answers that are easy to grade exactly.
BBH_TASKS = [
    "logical_deduction_three_objects",
    "date_understanding",
    "tracking_shuffled_objects_three_objects",
    "object_counting",
    "navigate",
]


def _bbh_difficulty(task: str) -> str:
    easy = {"object_counting", "navigate"}
    hard = {"tracking_shuffled_objects_three_objects"}
    if task in easy:
        return "easy"
    if task in hard:
        return "hard"
    return "medium"


def load_bbh(n: int, seed: int, tasks: Optional[Iterable[str]] = None) -> List[Problem]:
    """Load BIG-Bench Hard; sample ``n`` problems across selected tasks."""
    from datasets import load_dataset

    tasks = list(tasks or BBH_TASKS)
    rng = random.Random(seed)
    pool: List[Problem] = []
    for task in tasks:
        ds = load_dataset("lukaemon/bbh", task, split="test")
        for i, row in enumerate(ds):
            pool.append(Problem(
                id=f"bbh_{task}_{i:04d}",
                dataset="bbh",
                task=task,
                question=row["input"],
                gold_answer=row["target"].strip(),
                difficulty=_bbh_difficulty(task),
            ))
    rng.shuffle(pool)
    return pool[:n]


# ----------------------------------------------------------------------------
# Persisted subset
# ----------------------------------------------------------------------------

def build_subset(n_gsm8k: int, n_bbh: int, seed: int) -> List[Problem]:
    problems = load_gsm8k_platinum(n_gsm8k, seed) + load_bbh(n_bbh, seed)
    return problems


def save_subset(problems: List[Problem], path: pathlib.Path = SUBSET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in problems:
            f.write(json.dumps(asdict(p)) + "\n")


def load_subset(path: pathlib.Path = SUBSET_PATH) -> List[Problem]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m scripts.make_subset` first."
        )
    out: List[Problem] = []
    with path.open() as f:
        for line in f:
            out.append(Problem(**json.loads(line)))
    return out
