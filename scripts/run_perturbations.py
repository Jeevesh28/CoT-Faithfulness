"""Stage 3: apply four perturbations and re-ask the model for its answer only.

For each baseline row (problem, model) we produce four perturbed rows. To
re-inject the perturbed CoT we use ``prompts.build_reinject_messages``, which
passes the perturbed CoT inside the user turn and asks the model for only
the final answer on an ``Answer:`` line.

Output: ``results/perturbed_cots.jsonl``
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import traceback
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

from src.cot_parser import parse_cot
from src.evaluate import answer_equal
from src.models import get_client, load_config
from src.perturbations import (
    PERTURBATIONS, step_corruption, step_shuffling,
    early_termination, wrong_but_consistent,
)
from src.prompts import build_reinject_messages


def _apply_perturbation(ptype, cot_steps, *, seed, donor=None, keep_frac=0.5):
    """Dispatch to the right perturbation function with the right kwargs.

    This avoids ``early_termination()`` getting a stray ``seed=`` kwarg.
    """
    if ptype == "step_corruption":
        return step_corruption(cot_steps, seed=seed)
    if ptype == "step_shuffling":
        return step_shuffling(cot_steps, seed=seed)
    if ptype == "early_termination":
        return early_termination(cot_steps, keep_frac=keep_frac)
    if ptype == "wrong_but_consistent":
        if donor is None:
            raise ValueError("wrong_but_consistent requires donor")
        return wrong_but_consistent(cot_steps, donor)
    raise KeyError(ptype)


BASE = pathlib.Path("results/baseline_cots.jsonl")
OUT = pathlib.Path("results/perturbed_cots.jsonl")


def _load_baseline() -> List[dict]:
    with BASE.open() as f:
        return [json.loads(l) for l in f]


def _done_keys() -> set:
    if not OUT.exists():
        return set()
    done = set()
    with OUT.open() as f:
        for line in f:
            r = json.loads(line)
            done.add((r["id"], r["model"], r["perturbation"]))
    return done


def _donor_pool(baseline: List[dict]) -> Dict[str, List[List[str]]]:
    """For ``wrong_but_consistent``, we use valid CoTs from the same model on
    other problems. Only take well-formed correct CoTs so the donor is
    actually a plausible solution to *some* problem."""
    pool: Dict[str, List[List[str]]] = defaultdict(list)
    for r in baseline:
        if r["correct"] and r.get("well_formed", True) and r["cot_steps"]:
            pool[r["model"]].append(r["cot_steps"])
    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS.keys()))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    baseline = _load_baseline()
    print(f"Loaded {len(baseline)} baseline rows from {BASE}")

    donors = _donor_pool(baseline)
    done = _done_keys()

    # Process one model at a time so we never hold two big HF models in VRAM.
    models = sorted({r["model"] for r in baseline})
    print(f"Models to process (sequentially): {models}")

    with OUT.open("a") as fh:
        for model_name in models:
            print(f"\n=== {model_name} ===")
            client = get_client(model_name, cfg)
            mcfg = cfg["models"][model_name]
            rows_for_model = [r for r in baseline if r["model"] == model_name]
            for b in tqdm(rows_for_model, desc=model_name, file=sys.stdout):
                if not b["cot_steps"]:
                    continue
                rng = random.Random(args.seed + hash(b["id"]) % 2**31)

                for ptype in args.perturbations:
                    key = (b["id"], b["model"], ptype)
                    if key in done:
                        continue
                    try:
                        donor = None
                        if ptype == "wrong_but_consistent":
                            pool = [c for c in donors.get(b["model"], [])
                                    if c != b["cot_steps"]]
                            if not pool:
                                continue
                            donor = rng.choice(pool)
                        pert_steps, spans = _apply_perturbation(
                            ptype, b["cot_steps"],
                            seed=rng.randint(0, 2**31), donor=donor,
                        )

                        pert_text = "\n".join(pert_steps)
                        messages = build_reinject_messages(b["question"], pert_text)
                        g = client.generate(
                            messages,
                            max_new_tokens=min(128, mcfg.get("max_new_tokens", 128)),
                            temperature=mcfg.get("temperature", 0.0),
                        )
                        _, answer = parse_cot(g.text)
                        row = {
                            "id": b["id"],
                            "model": b["model"],
                            "perturbation": ptype,
                            "original_cot_steps": b["cot_steps"],
                            "perturbed_cot_steps": pert_steps,
                            "perturbed_char_spans": spans,
                            "perturbed_cot_raw": pert_text,
                            "question": b["question"],
                            "gold_answer": b["gold_answer"],
                            "predicted_answer": answer,
                            "raw_output": g.text,
                            "correct": answer_equal(answer, b["gold_answer"]),
                            "changed_answer": answer != b["predicted_answer"],
                            "tokens_used": g.tokens_used,
                        }
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                    except Exception as e:
                        print(f"  !! {key} failed: {e}")
                        traceback.print_exc()

            # Free GPU memory before loading the next HF model.
            del client
            try:
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
