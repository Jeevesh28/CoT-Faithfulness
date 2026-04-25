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

class HFChatClient(LLMClient):
    """Generic HuggingFace chat-instruct client used for BOTH Llama 3.1 8B and
    Gemma. Loads with ``attn_implementation="eager"`` so ``output_attentions``
    works for the attention analysis stage (only used for the Llama instance).

    Exposes ``self.model`` and ``self.tokenizer`` for ``src.attention``.
    """

    def __init__(self, cfg: dict, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.cfg = cfg

        model_id = cfg["model_id"]
        hf_token = os.environ.get("HF_TOKEN")
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        load_in_4bit = bool(cfg.get("load_in_4bit", False))

        load_kwargs: Dict[str, object] = dict(
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
            token=hf_token,
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs.pop("torch_dtype", None)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        last_err = None
        for candidate in [model_id, cfg.get("fallback_model_id")]:
            if not candidate:
                continue
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(candidate, token=hf_token)
                self.model = AutoModelForCausalLM.from_pretrained(candidate, **load_kwargs)
                self._loaded_model_id = candidate
                break
            except Exception as e:
                last_err = e
                print(f"[{name}] could not load {candidate}: {e}")
        else:
            raise RuntimeError(f"Failed to load {name}: {last_err}")

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _format(self, messages: List[Dict[str, str]]) -> str:
        # Gemma's chat template doesn't accept a separate "system" role, so
        # we merge any system messages into the first user turn.
        if "gemma" in self._loaded_model_id.lower():
            sys_text, merged = "", []
            for m in messages:
                if m["role"] == "system":
                    sys_text += m["content"] + "\n\n"
                elif m["role"] == "user":
                    merged.append({"role": "user",
                                   "content": (sys_text + m["content"]) if not merged else m["content"]})
                    sys_text = ""
                else:
                    merged.append(m)
            messages = merged
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


# Backwards-compatible alias used by ``src.attention``.
HFLlamaClient = HFChatClient


# ----------------------------------------------------------------------------
# Claude Opus 4.6
# ----------------------------------------------------------------------------

class AnthropicClient(LLMClient):

    def __init__(self, cfg: dict, name: str = "claude-opus-4-6"):
        import anthropic

        self.name = name
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

def get_client(name: str, cfg: Optional[dict] = None) -> LLMClient:
    cfg = cfg or load_config()
    if name not in cfg["models"]:
        raise KeyError(f"{name} not in configs/models.yaml")
    mcfg = cfg["models"][name]
    backend = mcfg["backend"]
    if backend == "hf":
        return HFChatClient(mcfg, name=name)
    if backend == "anthropic":
        return AnthropicClient(mcfg, name=name)
    raise KeyError(f"Unknown backend: {backend}")
