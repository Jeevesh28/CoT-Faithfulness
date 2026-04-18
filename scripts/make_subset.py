"""Build the frozen 75-problem split (50 GSM8K-Platinum + 25 BBH).

Commit ``data/processed/subset.jsonl`` so the experiment is reproducible.
"""
from __future__ import annotations

import argparse

from src.datasets import build_subset, save_subset, SUBSET_PATH
from src.models import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--out", default=str(SUBSET_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    print(f"Building subset: {data_cfg['gsm8k_n']} GSM8K + {data_cfg['bbh_n']} BBH "
          f"(seed={data_cfg['seed']})")

    problems = build_subset(
        n_gsm8k=data_cfg["gsm8k_n"],
        n_bbh=data_cfg["bbh_n"],
        seed=data_cfg["seed"],
    )
    save_subset(problems)
    print(f"Wrote {len(problems)} problems to {args.out}")
    from collections import Counter
    print("Difficulty counts:", Counter(p.difficulty for p in problems))


if __name__ == "__main__":
    main()
