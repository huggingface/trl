# Tiny model generation

This directory contains one script per tiny model used by the TRL test suite. Each script builds a random-weight, minimally-sized model on top of a real tokenizer/processor and pushes it to the `trl-internal-testing` organization on the Hub.

## Layout

```
generate_tiny_models/
├── _common.py                               # shared helpers (push_to_hub, smoke_test, print_config_diff, ...)
├── for_causal_lm/                # *ForCausalLM + GPT-2 LM head + small/PEFT variants
├── for_sequence_classification/  # *ForSequenceClassification (reward models)
└── for_conditional_generation/   # *ForConditionalGeneration (VLMs + T5 + Bart encoder-decoder)
```

## Running

From the repo root, invoke a script by its module path:

```bash
python -m scripts.generate_tiny_models.for_causal_lm.qwen3_for_causal_lm
```

Each script:

1. Checks that the installed `transformers` version matches the one pinned in the script (fails otherwise).
2. Builds the tiny model with random weights.
3. Runs `smoke_test` — a minimal forward pass to catch config misspecification and NaNs.
4. Runs `print_config_diff` — prints every flat-key difference between the reference Hub config and the tiny model's config (for debugging scale-downs).
5. Pushes the model, tokenizer/processor, generation config, and model card to the Hub.

If the repo already exists on the Hub, the push is skipped (pass `force=True` in `push_to_hub(...)` to overwrite).

## Version pinning

Every script declares `TRANSFORMERS_VERSION = "X.Y.Z"`, which is:

```
max(version that introduced the model, TRL's transformers floor)
```

The floor (currently `4.56.2`) is the `transformers>=` bound from `pyproject.toml`. Scripts for models introduced after the floor pin a higher version (e.g. Qwen3-VL pins `4.57.0`, Gemma4 pins `5.6.0`). The check is an exact match via `packaging.version.Version`; install the pinned version before running.

**Why exact?** transformers is backward-compatible (a checkpoint saved by X loads on any ≥ X) but not forward-compatible. TRL CI runs against the floor, so tiny models must be saved with the oldest version that supports them — any newer save risks using config fields the floor can't parse. The exact-match check prevents accidental drift.

## Adding a new tiny model

1. Pick the right subfolder based on the model class suffix (`ForCausalLM`, `ForSequenceClassification`, `ForConditionalGeneration`).
2. Copy an existing script with the closest shape and adapt it — reference model id, config class, model class, special kwargs.
3. Set `TRANSFORMERS_VERSION` to the release that introduced the model (or to the TRL floor, whichever is higher).
4. Run it. Inspect the `[smoke_test]` and `[config_diff]` output before letting it push.
