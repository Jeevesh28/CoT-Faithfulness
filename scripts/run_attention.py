"""Stage 5: attention analysis on Llama 3.1 8B.

Selects a 30-problem subset (10 behaviorally unfaithful, 10 faithful, 10 hard),
runs a teacher-forced forward pass for the *perturbed* CoT of each selected
case, and saves a CoT attention ratio + per-token attention for heatmaps.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.attention import compute_attention, save_attention
from src.models import get_client, load_config


BASE = pathlib.Path("results/baseline_cots.jsonl")
PERT = pathlib.Path("results/perturbed_cots.jsonl")
OUT_DIR = pathlib.Path("results/attention")
SUBSET_JSON = OUT_DIR / "_subset.json"


def _load(path):
    with path.open() as f:
        return [json.loads(l) for l in f]


def select_subset(baseline, perturbed, n_unfaithful=10, n_faithful=10, n_hard=10, seed=0) -> List[Dict]:
    """Pick 30 (problem, perturbation) pairs on the Llama model."""
    rng = random.Random(seed)
    base_by = {(r["id"], r["model"]): r for r in baseline}
    pert = [r for r in perturbed if r["model"] == "llama-3.1-8b"]

    # Filter to cases where we have a baseline row with a parsed answer.
    enriched = []
    for r in pert:
        b = base_by.get((r["id"], r["model"]))
        if not b:
            continue
        enriched.append({**r,
                         "baseline_correct": b["correct"],
                         "difficulty": b["difficulty"],
                         "baseline_predicted_answer": b["predicted_answer"]})

    # Unfaithful: baseline correct + perturbed still correct (CoT ignored).
    unfaithful = [r for r in enriched
                  if r["baseline_correct"] and r["correct"] and not r["changed_answer"]]
    # Faithful: baseline correct + perturbed incorrect (CoT mattered).
    faithful = [r for r in enriched
                if r["baseline_correct"] and not r["correct"] and r["changed_answer"]]
    # Hard: baseline incorrect overall (difficulty="hard" OR baseline wrong).
    hard = [r for r in enriched if r["difficulty"] == "hard"]

    def sample(lst, k):
        rng.shuffle(lst)
        return lst[:k]

    chosen = (sample(unfaithful, n_unfaithful)
              + sample(faithful, n_faithful)
              + sample(hard, n_hard))
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--model", default="llama-3.1-8b")
    parser.add_argument("--store-full", action="store_true",
                        help="Save full answer→context attention for heatmaps.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    acfg = cfg["attention"]

    baseline = _load(BASE)
    perturbed = _load(PERT)
    subset = select_subset(baseline, perturbed,
                           n_unfaithful=acfg["n_unfaithful"],
                           n_faithful=acfg["n_faithful"],
                           n_hard=acfg["n_hard"])

    SUBSET_JSON.write_text(json.dumps(
        [{"id": r["id"], "perturbation": r["perturbation"]} for r in subset], indent=2))
    print(f"Selected {len(subset)} cases → {SUBSET_JSON}")

    client = get_client(args.model, cfg)

    scatter_rows = []
    for row in tqdm(subset, desc="attention", file=sys.stdout):
        result = compute_attention(
            client,
            question=row["question"],
            cot_text=row["perturbed_cot_raw"],
            answer_text=row["predicted_answer"] or "",
            perturbed_char_spans=row.get("perturbed_char_spans") or None,
            store_full_attn=args.store_full,
        )
        fname = OUT_DIR / f"{row['id']}__{row['perturbation']}.npz"
        save_attention(fname, result)
        scatter_rows.append({
            "id": row["id"],
            "perturbation": row["perturbation"],
            "difficulty": row["difficulty"],
            "baseline_correct": row["baseline_correct"],
            "changed_answer": row["changed_answer"],
            "cot_attention_ratio": result.get("cot_attention_ratio"),
            "perturbed_attention_ratio": result.get("perturbed_attention_ratio"),
            "n_answer_tokens": result.get("n_answer_tokens"),
        })

    (OUT_DIR / "scatter.json").write_text(json.dumps(scatter_rows, indent=2))
    print(f"Wrote per-case npz files + scatter.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
