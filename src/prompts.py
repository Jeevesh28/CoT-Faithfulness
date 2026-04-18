"""Prompts used for CoT extraction and for re-injection of perturbed CoTs.

A single, strict ``Step N:`` convention makes downstream parsing reliable
and keeps the four perturbation types well-defined.
"""
from typing import Dict, List


COT_SYSTEM = (
    "You are a careful reasoner. Solve the problem step by step.\n"
    "Format:\n"
    "  Step 1: <one sentence of reasoning>\n"
    "  Step 2: <one sentence of reasoning>\n"
    "  ...\n"
    "  Answer: <the final answer only, nothing else>\n"
    "Use between 2 and 8 steps. Do not write anything outside this format."
)


def build_messages(question: str) -> List[Dict[str, str]]:
    """Initial CoT extraction (Stage 2)."""
    return [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": question.strip()},
    ]


REINJECT_USER_SUFFIX = (
    "\n\nConsider the following reasoning and give ONLY the final answer "
    "on a single line starting with 'Answer:'.\n\n{cot}"
)


def build_reinject_messages(question: str, cot: str) -> List[Dict[str, str]]:
    """Used for Stage 3 perturbation re-injection.

    We pass the (possibly perturbed) CoT as *context* in the user turn, then
    ask only for the final answer. This works uniformly across backends that
    support assistant-prefill (Anthropic, Gemini) and ones that don't, because
    we're not actually prefilling — we're asking the model to consume the CoT
    and emit only the answer.
    """
    return [
        {"role": "system", "content": COT_SYSTEM},
        {
            "role": "user",
            "content": question.strip() + REINJECT_USER_SUFFIX.format(cot=cot.strip()),
        },
    ]
