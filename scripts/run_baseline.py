"""Stage 2: baseline CoT extraction.

For each (problem, model) pair, run the model under the strict CoT prompt,
parse out the labeled steps + answer, and score against the gold.

Output: ``results/baseline_cots.jsonl``
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import traceback

from tqdm import tqdm

from src.cot_parser import parse_cot, is_well_formed
from src.datasets import load_subset
from src.evaluate import answer_equal
from src.models import get_client, load_config
from src.prompts import build_messages


OUT = pathlib.Path("results/baseline_cots.jsonl")


def _already_done(model: str) -> set:
    if not OUT.exists():
        return set()
    done = set()
    with OUT.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("model") == model:
                done.add(r["id"])
    return done


def run_model(model_name: str, cfg: dict) -> None:
    print(f"\n=== {model_name} ===")
    client = get_client(model_name, cfg)
    mcfg = cfg["models"][model_name]
    problems = load_subset()
    done_ids = _already_done(model_name)
    remaining = [p for p in problems if p.id not in done_ids]
    print(f"  {len(problems) - len(remaining)} already done; {len(remaining)} to run")

    with OUT.open("a") as fh:
        for prob in tqdm(remaining, desc=model_name, file=sys.stdout):
            try:
                g = client.generate(
                    build_messages(prob.question),
                    max_new_tokens=mcfg.get("max_new_tokens", 1024),
                    temperature=mcfg.get("temperature", 0.0),
                )
                steps, answer = parse_cot(g.text)
                row = {
                    "id": prob.id,
                    "dataset": prob.dataset,
                    "model": model_name,
                    "difficulty": prob.difficulty,
                    "question": prob.question,
                    "gold_answer": prob.gold_answer,
                    "cot_raw": g.text,
                    "cot_steps": steps,
                    "predicted_answer": answer,
                    "correct": answer_equal(answer, prob.gold_answer),
                    "well_formed": is_well_formed(steps, answer),
                    "tokens_used": g.tokens_used,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            except Exception as e:
                print(f"  !! {prob.id} failed: {e}")
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--models", nargs="+",
                        default=["llama-3.1-8b", "gemma-3-12b", "claude-opus-4-6"])
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    for m in args.models:
        run_model(m, cfg)
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
