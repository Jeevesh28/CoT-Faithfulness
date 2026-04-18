"""Stage 4: macro + micro analysis. Produces:
  * results/faithfulness_table.csv           (macro)
  * results/by_difficulty.csv                (macro, hypothesis 1)
  * results/figures/faithfulness_*.png       (heatmaps)
  * results/figures/convergence_scatter.png  (stage 5 headline)
  * results/figures/micro_<id>.png           (three case-study cards)
  * results/figures/attention_<id>_<pert>.png (heatmaps from stored .npz)
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd

from src.evaluate import build_summary, by_difficulty
from src import viz


BASE = pathlib.Path("results/baseline_cots.jsonl")
PERT = pathlib.Path("results/perturbed_cots.jsonl")
ATTN = pathlib.Path("results/attention")
FIG = pathlib.Path("results/figures")


def _load_jsonl(p: pathlib.Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in p.open()]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-n", type=int, default=3,
                        help="Number of micro case-study figures to render.")
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    base = _load_jsonl(BASE)
    pert = _load_jsonl(PERT)
    print(f"Baseline rows: {len(base)}   Perturbed rows: {len(pert)}")

    # --- Macro summary ----------------------------------------------------
    summary = build_summary(base, pert)
    summary.to_csv("results/faithfulness_table.csv", index=False)
    print("\n=== Faithfulness table ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary.round(3).to_string(index=False))

    for ds in summary["dataset"].unique():
        viz.plot_faithfulness_heatmap(summary, ds, FIG / f"faithfulness_{ds}.png")

    diff_df = by_difficulty(base, pert)
    diff_df.to_csv("results/by_difficulty.csv", index=False)
    viz.plot_f_by_difficulty(diff_df, FIG / "f_by_difficulty.png")

    # --- Stage 5 convergence scatter (if attention/scatter.json exists) ---
    scatter_path = ATTN / "scatter.json"
    if scatter_path.exists():
        scatter_df = pd.DataFrame(json.loads(scatter_path.read_text()))
        if len(scatter_df):
            viz.plot_convergence(scatter_df, FIG / "convergence_scatter.png")
            print(f"Wrote convergence scatter (n={len(scatter_df)})")

    # --- Micro case studies (3 by default) --------------------------------
    # Pick interesting cases: at least one where >=1 model is unfaithful
    # and >=1 model is faithful (answer changes) under step_corruption.
    merged = pert.merge(
        base[["id", "model", "correct"]].rename(columns={"correct": "baseline_correct"}),
        on=["id", "model"])
    sc = merged[merged["perturbation"] == "step_corruption"]
    by_id = sc.groupby("id")["changed_answer"].agg(["sum", "count"]).reset_index()
    split_cases = by_id[(by_id["sum"] > 0) & (by_id["sum"] < by_id["count"])]
    chosen_ids = split_cases.head(args.micro_n)["id"].tolist()
    print(f"Micro cases: {chosen_ids}")

    for pid in chosen_ids:
        # Gather baseline + 4 perturbations across all models.
        rows_per_model: Dict[str, List[dict]] = {}
        for m in base["model"].unique():
            row_base = base[(base["id"] == pid) & (base["model"] == m)]
            rows_per_model[m] = []
            if len(row_base):
                r = row_base.iloc[0].to_dict()
                r["perturbation"] = "baseline"
                r["perturbed_cot_raw"] = r["cot_raw"]
                rows_per_model[m].append(r)
            for _, prow in pert[(pert["id"] == pid) & (pert["model"] == m)].iterrows():
                rows_per_model[m].append(prow.to_dict())
        problem = base[base["id"] == pid].iloc[0].to_dict()
        viz.micro_card(problem, rows_per_model, FIG / f"micro_{pid}.png")

    # --- Attention heatmaps for stored full-attention .npz ----------------
    for npz_path in ATTN.glob("*.npz"):
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "attn_from_answer" not in data.files:
                continue
            viz.plot_attention_heatmap(
                tokens=list(data["token_strings"]),
                attn_from_ans=data["attn_from_answer"],
                cot_mask=data["cot_mask"],
                pert_mask=data["perturbed_mask"],
                out_path=FIG / f"attention_{npz_path.stem}.png",
            )
        except Exception as e:
            print(f"Skipping heatmap for {npz_path.stem}: {e}")

    print(f"\nAll figures in {FIG}")


if __name__ == "__main__":
    main()
