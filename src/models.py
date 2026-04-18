"""Unified LLMClient abstraction over three backends.

* ``HFLlamaClient`` — Llama 3.1 8B via HuggingFace ``transformers``.
  This is the client used for both text generation *and* attention analysis
  (via the ``model`` attribute exposed for ``src.attention``).
* ``GeminiClient`` — Google Gemini 2.5 Flash Lite via ``google-generativeai``.
* ``AnthropicClient`` — Claude Sonnet 4.5 via ``anthropic``.

All three expose ``generate(messages, max_new_tokens, temperature)``.
"""
from __future__ import annotations

import os
import time
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


# ----------------------------------------------------------------------------
# Generation record
# ----------------------------------------------------------------------------

@dataclass
class Generation:
    text: str                  # the model's output (NOT including any prefix)
    tokens_used: int           # total tokens, best-effort across backends
    raw: Dict[str, Any]        # raw backend response (for debugging)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

def load_config(path: str = "configs/models.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------------

class LLMClient(ABC):
    """Every backend exposes the same ``generate`` signature."""

    name: str

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Generation: ...


# ----------------------------------------------------------------------------
# HuggingFace Llama 3.1 client
# ----------------------------------------------------------------------------

class HFLlamaClient(LLMClient):
    """Llama 3.1 8B Instruct via ``transformers``.

    On an A100 (bf16) this uses ~16 GB VRAM and runs a forward pass in <1s
    on typical CoT-length inputs. We expose ``self.model`` and ``self.tokenizer``
    so the attention module can reuse the loaded checkpoint.
    """

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = "llama-3.1-8b"
        self.cfg = cfg

        model_id = cfg["model_id"]
        hf_token = os.environ.get("HF_TOKEN")
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="eager",   # needed for output_attentions
                token=hf_token,
            )
        except Exception:
            # Fallback to ungated mirror if the user has no license acceptance.
            fallback = cfg.get("fallback_model_id")
            if not fallback:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="eager",
            )

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _format(self, messages: List[Dict[str, str]]) -> str:
        # Llama-3.1 has a native chat template.
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, messages, max_new_tokens=1024, temperature=0.0) -> Generation:
        import torch

        prompt = self._format(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-5),
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return Generation(
            text=text,
            tokens_used=int(out.shape[1]),
            raw={"prompt_tokens": int(inputs["input_ids"].shape[1])},
        )


# ----------------------------------------------------------------------------
# Gemini 2.5 Flash Lite
# ----------------------------------------------------------------------------

class GeminiClient(LLMClient):

    def __init__(self, cfg: dict):
        import google.generativeai as genai

        self.name = "gemini-2.5-flash-lite"
        self.cfg = cfg
        api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model_id = cfg["model_id"]

    @staticmethod
    def _to_gemini(messages):
        """Gemini expects {role: user|model, parts: [text]}. System prompts
        are merged into the first user message."""
        system = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n\n"
            elif m["role"] == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": (system + m["content"]) if not contents else m["content"]}],
                })
                system = ""
            elif m["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": m["content"]}]})
        return contents

    def generate(self, messages, max_new_tokens=1024, temperature=0.0) -> Generation:
        contents = self._to_gemini(messages)
        model = self.genai.GenerativeModel(self.model_id)
        for attempt in range(4):
            try:
                resp = model.generate_content(
                    contents,
                    generation_config={
                        "max_output_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                )
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        text = resp.text if getattr(resp, "text", None) else ""
        usage = getattr(resp, "usage_metadata", None)
        tokens = int(getattr(usage, "total_token_count", 0) or 0)
        return Generation(text=text, tokens_used=tokens, raw={"candidates": len(resp.candidates)})


# ----------------------------------------------------------------------------
# Claude Sonnet 4.5
# ----------------------------------------------------------------------------

class AnthropicClient(LLMClient):

    def __init__(self, cfg: dict):
        import anthropic

        self.name = "claude-sonnet-4-5"
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        self.model_id = cfg["model_id"]

    @staticmethod
    def _split_system(messages):
        system_parts, msgs = [], []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        return ("\n\n".join(system_parts).strip() or None), msgs

    def generate(self, messages, max_new_tokens=1024, temperature=0.0) -> Generation:
        system, msgs = self._split_system(messages)
        for attempt in range(4):
            try:
                r = self.client.messages.create(
                    model=self.model_id,
                    system=system,
                    messages=msgs,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        tokens = r.usage.input_tokens + r.usage.output_tokens
        return Generation(text=text, tokens_used=tokens, raw={"stop": r.stop_reason})


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------

_BACKENDS = {
    "hf": HFLlamaClient,
    "gemini": GeminiClient,
    "anthropic": AnthropicClient,
}


def get_client(name: str, cfg: Optional[dict] = None) -> LLMClient:
    cfg = cfg or load_config()
    if name not in cfg["models"]:
        raise KeyError(f"{name} not in configs/models.yaml")
    mcfg = cfg["models"][name]
    return _BACKENDS[mcfg["backend"]](mcfg)
