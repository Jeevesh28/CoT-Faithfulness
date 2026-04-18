"""Plotting: macro heatmaps, attention heatmaps, micro side-by-side cards,
and the CoT-attention-vs-behavior scatter."""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ----------------------------------------------------------------------------
# Macro faithfulness heatmap
# ----------------------------------------------------------------------------

def plot_faithfulness_heatmap(summary: pd.DataFrame, dataset: str,
                              out_path: pathlib.Path) -> None:
    d = summary[summary["dataset"] == dataset]
    pivot = d.pivot(index="model", columns="perturbation", values="F")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0, vmax=1,
                cmap="viridis", ax=ax, cbar_kws={"label": "F (faithfulness)"})
    ax.set_title(f"Faithfulness score F — {dataset}")
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Model")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_f_by_difficulty(diff_df: pd.DataFrame, out_path: pathlib.Path) -> None:
    g = sns.catplot(data=diff_df, x="difficulty", y="F", hue="perturbation",
                    col="model", kind="bar", sharey=True, height=3.5, aspect=1.1,
                    order=["easy", "medium", "hard"])
    g.set_axis_labels("Difficulty", "F")
    g.fig.suptitle("Faithfulness by difficulty (Hypothesis 1)", y=1.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=160)
    plt.close(g.fig)


# ----------------------------------------------------------------------------
# Attention heatmap (token-level, for a single case study)
# ----------------------------------------------------------------------------

def plot_attention_heatmap(tokens: List[str], attn_from_ans: np.ndarray,
                           cot_mask: np.ndarray, pert_mask: np.ndarray,
                           out_path: pathlib.Path,
                           max_tokens: int = 120) -> None:
    """Heatmap of attention from answer tokens to context tokens. We only
    plot the last ``max_tokens`` tokens to keep the figure readable."""
    T = attn_from_ans.shape[1]
    start = max(0, T - max_tokens)
    slice_ = slice(start, T)
    att = attn_from_ans[:, slice_]
    tok = tokens[slice_]

    fig, ax = plt.subplots(figsize=(min(24, 0.18 * len(tok) + 2), 2.5 + 0.2 * att.shape[0]))
    sns.heatmap(att, ax=ax, cmap="magma", cbar_kws={"label": "attention"})
    ax.set_xticks(np.arange(len(tok)) + 0.5)
    ax.set_xticklabels(tok, rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("context tokens (last window)")
    ax.set_title("Attention from answer tokens → context")

    # Overlay markers for CoT (blue) and perturbed (red) spans along x-axis.
    for i in range(len(tok)):
        t_idx = start + i
        if pert_mask[t_idx]:
            ax.axvspan(i, i + 1, color="red", alpha=0.18)
        elif cot_mask[t_idx]:
            ax.axvspan(i, i + 1, color="blue", alpha=0.08)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Convergence scatter (Stage 5 headline figure)
# ----------------------------------------------------------------------------

def plot_convergence(scatter_df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """scatter_df columns: id, perturbation, changed_answer (bool),
    cot_attention_ratio, perturbed_attention_ratio."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for pert, grp in scatter_df.groupby("perturbation"):
        ax.scatter(grp["cot_attention_ratio"], grp["changed_answer"].astype(int),
                   alpha=0.6, s=60, label=pert)
    ax.set_xlabel("CoT attention ratio (mechanistic)")
    ax.set_ylabel("Changed answer under perturbation (behavioral)")
    ax.set_title("Mechanistic vs. behavioral faithfulness — Llama 3.1 8B")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["unfaithful (no change)", "faithful (changed)"])
    ax.legend(title="Perturbation", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Micro side-by-side card (as a figure, not a webpage — for write-up inclusion)
# ----------------------------------------------------------------------------

def micro_card(problem: Dict, rows_per_model: Dict[str, List[Dict]],
               out_path: pathlib.Path) -> None:
    """Render one problem with its baseline + 4 perturbed CoTs across all
    models, side by side, as a matplotlib figure of text blocks."""
    models = list(rows_per_model.keys())
    pert_types = ["baseline", "step_corruption", "wrong_but_consistent",
                  "early_termination", "step_shuffling"]
    fig, axes = plt.subplots(len(models), len(pert_types),
                             figsize=(4 * len(pert_types), 3.2 * len(models)))
    if len(models) == 1:
        axes = np.array([axes])
    for r, model in enumerate(models):
        for c, pt in enumerate(pert_types):
            ax = axes[r, c]
            ax.axis("off")
            row = next((x for x in rows_per_model[model] if x.get("perturbation", "baseline") == pt), None)
            title = f"{model}\n{pt}"
            if row is None:
                ax.set_title(title, fontsize=9)
                continue
            cot = row.get("perturbed_cot_raw") or row.get("cot_raw") or ""
            if len(cot) > 420:
                cot = cot[:420] + "…"
            correct = row.get("correct", False)
            pred = row.get("predicted_answer", "?")
            header_color = "green" if correct else "crimson"
            ax.set_title(f"{title}\npred={pred}  correct={correct}",
                         fontsize=9, color=header_color)
            ax.text(0, 1, cot, fontsize=7, family="monospace",
                    verticalalignment="top")
    fig.suptitle(f"Q: {problem['question'][:120]}…  |  gold={problem['gold_answer']}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
