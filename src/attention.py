"""Attention extraction for the open-weight Llama 3.1 model.

Given a (question, CoT, answer) triple, we teacher-force the full conversation
and read attention from the answer tokens back to the context. We compute:

  * ``cot_attention_ratio`` — fraction of answer-token attention mass that
    lands on the CoT span (as opposed to the question, system prompt, or
    special tokens).
  * ``perturbed_attention_ratio`` — optional, fraction of that attention
    mass that lands specifically on the *perturbed* tokens.

This module assumes a loaded HFLlamaClient from ``src.models``.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.prompts import COT_SYSTEM


# ----------------------------------------------------------------------------
# Prompt assembly with span tracking
# ----------------------------------------------------------------------------

def _build_conversation(tokenizer, question: str, cot_text: str, answer_text: str):
    """Build the full chat string and return it + a 'template' splitter.

    We apply the chat template in two stages so we know exactly which
    substring is the CoT (for masking) and which is the answer (for
    reading attention *from*).
    """
    user_msg = {"role": "user", "content": question}
    prefix_msgs = [{"role": "system", "content": COT_SYSTEM}, user_msg]
    # First: just the system+user to get the generation prompt prefix.
    prefix = tokenizer.apply_chat_template(
        prefix_msgs, tokenize=False, add_generation_prompt=True
    )
    # Second: append the assistant turn as plain text.
    assistant_body = f"{cot_text}\nAnswer: {answer_text}"
    full = prefix + assistant_body
    return full, prefix, assistant_body, cot_text


def _span_token_mask(offsets: List[Tuple[int, int]], char_start: int, char_end: int) -> np.ndarray:
    """Boolean mask of tokens whose char range overlaps [char_start, char_end)."""
    m = np.zeros(len(offsets), dtype=bool)
    for i, (a, b) in enumerate(offsets):
        if b <= char_start or a >= char_end:
            continue
        if a == b == 0:  # special / pad tokens have (0, 0)
            continue
        m[i] = True
    return m


# ----------------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------------

@torch.no_grad()
def compute_attention(
    client,                               # HFLlamaClient
    question: str,
    cot_text: str,
    answer_text: str,
    perturbed_char_spans: Optional[List[Tuple[int, int]]] = None,
    store_full_attn: bool = False,
) -> Dict:
    """Run a forward pass and compute CoT attention ratios.

    Returns a dict with floats (ratios) and optionally the full
    ``[L, H, n_ans, T]`` attention tensor (if ``store_full_attn=True``).
    """
    tok = client.tokenizer
    mdl = client.model

    full, prefix, assistant_body, cot_part = _build_conversation(
        tok, question, cot_text, answer_text
    )

    enc = tok(full, return_offsets_mapping=True, return_tensors="pt",
              add_special_tokens=False)
    input_ids = enc["input_ids"].to(mdl.device)
    offsets = enc["offset_mapping"][0].tolist()

    # Locate CoT and answer spans in char offsets of `full`.
    cot_start = len(prefix)
    cot_end = cot_start + len(cot_part)
    ans_label = "\nAnswer: "
    ans_char_start = cot_end + len(ans_label)
    ans_char_end = len(full)

    cot_mask = _span_token_mask(offsets, cot_start, cot_end)
    ans_mask = _span_token_mask(offsets, ans_char_start, ans_char_end)

    pert_mask = np.zeros(len(offsets), dtype=bool)
    if perturbed_char_spans:
        # Spans are relative to the CoT (starting at 0), so shift by cot_start.
        for a, b in perturbed_char_spans:
            pert_mask |= _span_token_mask(offsets, cot_start + a, cot_start + b)

    # Run the model with output_attentions.
    out = mdl(input_ids, output_attentions=True, use_cache=False)
    # attentions: tuple length L, each [B, H, T, T]
    attn = torch.stack(out.attentions, dim=0)       # [L, 1, H, T, T]
    attn = attn.squeeze(1).float().cpu().numpy()    # [L, H, T, T]

    # Aggregate: mean over layers + heads, then restrict to answer-token rows.
    attn_mean = attn.mean(axis=(0, 1))              # [T, T]
    ans_idx = np.where(ans_mask)[0]
    if ans_idx.size == 0:
        return {
            "cot_attention_ratio": float("nan"),
            "perturbed_attention_ratio": float("nan") if perturbed_char_spans else None,
            "n_answer_tokens": 0,
        }

    attn_from_ans = attn_mean[ans_idx]              # [n_ans, T]
    total = attn_from_ans.sum() + 1e-12
    cot_mass = attn_from_ans[:, cot_mask].sum()
    cot_ratio = float(cot_mass / total)

    result = {
        "cot_attention_ratio": cot_ratio,
        "n_answer_tokens": int(ans_idx.size),
        "n_cot_tokens": int(cot_mask.sum()),
        "n_total_tokens": int(len(offsets)),
    }
    if perturbed_char_spans:
        pert_mass = attn_from_ans[:, pert_mask].sum()
        result["perturbed_attention_ratio"] = float(pert_mass / total)
        result["n_perturbed_tokens"] = int(pert_mask.sum())

    if store_full_attn:
        result["attn_from_answer"] = attn_from_ans      # [n_ans, T]
        result["token_strings"] = [tok.decode([int(t)]) for t in input_ids[0].cpu().numpy()]
        result["cot_mask"] = cot_mask
        result["perturbed_mask"] = pert_mask
        result["answer_mask"] = ans_mask

    return result


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def save_attention(path: pathlib.Path, result: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saveable = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            saveable[k] = v
        elif isinstance(v, list):
            # token_strings etc.
            saveable[k] = np.array(v, dtype=object)
        else:
            saveable[k] = np.array(v)
    np.savez_compressed(path, **saveable)
