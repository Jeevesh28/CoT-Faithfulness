# CoT Faithfulness

Implementation for *"Is the Chain-of-Thought Actually Thinking?"* — CS 639, Spring 2026.
Team: Jeevesh Mahajan, Saksham Garg, Rohan Gupta.

We measure **chain-of-thought faithfulness** across three models:

| Slot | Model | Backend | Used for |
|------|-------|---------|----------|
| Open-weight | Llama 3.1 8B Instruct | HuggingFace `transformers` | Behavioral **and** attention |
| Closed (small) | Gemini 2.5 Flash Lite | Google AI Studio | Behavioral only |
| Closed (big) | Claude Sonnet 4.5 | Anthropic API | Behavioral only |

## Quickstart (Colab A100 recommended)

```bash
git clone <this repo>
cd cot-faithfulness
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN

# 1. Prepare 75-problem split (50 GSM8K-Platinum + 25 BBH)
python -m scripts.make_subset

# 2. Stage 2 — baseline CoT extraction (225 rows: 75 × 3 models)
python -m scripts.run_baseline --models llama-3.1-8b gemini-2.5-flash-lite claude-sonnet-4-5

# 3. Stage 3 — four perturbations × 3 models (900 rows)
python -m scripts.run_perturbations

# 4. Stage 5 — attention extraction on Llama (30-problem subset)
python -m scripts.run_attention

# 5. Stage 4 — macro + micro analysis
python -m scripts.analyze
jupyter notebook notebooks/results.ipynb
```

## Repository Layout

```
cot-faithfulness/
├── configs/models.yaml        # model registry + dataset sizes
├── src/
│   ├── datasets.py            # GSM8K-Platinum + BBH loaders
│   ├── models.py              # LLMClient abstraction (HF / Gemini / Anthropic)
│   ├── prompts.py             # CoT prompt template
│   ├── cot_parser.py          # text → labeled steps + answer
│   ├── perturbations.py       # four perturbation functions
│   ├── evaluate.py            # faithfulness score + bootstrap CIs
│   ├── attention.py           # Llama attention extraction
│   └── viz.py                 # heatmaps, micro cards, scatter
├── scripts/                   # one entry point per stage
├── tests/                     # pytest unit tests
├── notebooks/results.ipynb    # final figures
├── data/processed/            # frozen 75-problem split
└── results/                   # baseline/perturbed jsonl, attention npz, figures
```

## Data Contracts

**`results/baseline_cots.jsonl`** (one row per problem × model)
```json
{"id": "gsm8k_0042", "dataset": "gsm8k_platinum", "model": "llama-3.1-8b",
 "difficulty": "medium", "question": "...", "gold_answer": "18",
 "cot_raw": "Step 1: ...\nAnswer: 18", "cot_steps": ["Step 1: ...", ...],
 "predicted_answer": "18", "correct": true, "tokens_used": 184}
```

**`results/perturbed_cots.jsonl`** (one row per problem × model × perturbation)
```json
{"id": "gsm8k_0042", "model": "llama-3.1-8b", "perturbation": "step_corruption",
 "original_cot_steps": [...], "perturbed_cot_steps": [...],
 "perturbed_char_spans": [[12, 14]], "perturbed_cot_raw": "...",
 "predicted_answer": "18", "correct": true, "changed_answer": false}
```

## Cost Estimate

| Model | Calls | Approx. cost |
|-------|-------|-------------|
| Llama 3.1 8B (HF, A100 on Colab) | 375 | ~20 min of A100 time |
| Gemini 2.5 Flash Lite | 375 | Free tier (~25 min @ 15 RPM) |
| Claude Sonnet 4.5 | 375 | ~$3–$6 |

## Role Split (40-hour budget, ~13h / person)

- **Saksham** — Track A: `datasets.py`, `models.py` (Gemini + Claude clients), `prompts.py`, `cot_parser.py`, `run_baseline.py`
- **Rohan** — Track B: `perturbations.py`, `evaluate.py`, `run_perturbations.py`, macro analysis in `analyze.py`
- **Jeevesh** — Track C: `models.py` (HF Llama client), `attention.py`, `viz.py`, `run_attention.py`, `results.ipynb`, report integration

## Reproducibility

- Seeds are set in `configs/models.yaml`.
- The 75-problem split is frozen in `data/processed/subset.jsonl` — commit this file.
- All API calls use `temperature=0.0`.
- All scripts checkpoint to JSONL and resume on re-run.
