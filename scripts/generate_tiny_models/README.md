# Tiny model generation

This directory contains one script per tiny model used by the TRL test suite. Each script builds a random-weight, minimally-sized model on top of a real tokenizer/processor and pushes it to the `trl-internal-testing` organization on the Hub.

## Layout

```
generate_tiny_models/
├── _common.py                    # shared helpers (push_to_hub, smoke_test, print_config_diff, ...)
├── for_causal_lm/                # *ForCausalLM + GPT-2 LM head + small/PEFT variants
├── for_sequence_classification/  # *ForSequenceClassification (reward models)
├── for_conditional_generation/   # *ForConditionalGeneration (VLMs + T5 + Bart encoder-decoder)
└── for_block_diffusion/          # *ForBlockDiffusion (DiffusionGemma)
```

## Running

From the repo root, invoke a script by its module path:

```bash
python -m scripts.generate_tiny_models.for_causal_lm.qwen3_for_causal_lm
```

Each script:

1. Checks that the installed `transformers` version matches the one pinned in the script (fails otherwise).
2. Calls `set_seed` to seed the RNGs that model construction draws from.
3. Builds the tiny model with random weights.
4. Runs `smoke_test` — a minimal forward pass to catch config misspecification and NaNs.
5. Runs `check_dtype_pattern` — reads the reference safetensors header via the Hub API and flags any tensor whose dtype diverges from the reference (catches e.g. fp32 norms kept inside a bf16 checkpoint).
6. Runs `print_config_diff` — prints every flat-key difference between the reference Hub config and the tiny model's config (for debugging scale-downs).
7. Pushes the model, tokenizer/processor, generation config, and model card to the Hub in a single commit.

If the repo already exists on the Hub, the push is skipped by default. Pass `--create-pr` to open a PR against the existing repo instead:

```bash
python -m scripts.generate_tiny_models.for_causal_lm.qwen3_for_causal_lm --create-pr
```

Direct overwrites of `main` aren't supported — update via `--create-pr` and merge the PR on the Hub.

## Version pinning

Each script calls `check_transformers_version(...)`, which enforces:

```
max(version that introduced the model, TRL's transformers floor)
```

By default — `check_transformers_version()` with no argument — the expected version is the `transformers>=` floor read from `pyproject.toml` (currently `4.56.2`). Scripts for models introduced after the floor pass an explicit pin, e.g. `check_transformers_version("4.57.0")` for Qwen3-VL or `check_transformers_version("5.6.0")` for Gemma4. The check is an exact match via `packaging.version.Version`; install the matching version before running.

**Why exact?** transformers is backward-compatible (a checkpoint saved by X loads on any ≥ X) but not forward-compatible. TRL CI runs against the floor, so tiny models must be saved with the oldest version that supports them — any newer save risks using config fields the floor can't parse. The exact-match check prevents accidental drift.

## Seeding

Every script calls `set_seed()` right after the version check, before the model is built. The shared `SEED` lives in `_common.py`.

This matters because `HfApi.create_commit` drops files that are unchanged, so a re-push of an unmodified tiny model should be a no-op and open no PR. Without a seed the weights differ on every run, so `model.safetensors` always changed and every regeneration produced a diff whether or not anything real had changed. Seeding also makes a config-only change produce a config-only Hub diff.

The call has to come before the model is constructed rather than before `init_weights_tiny_model`: most scripts never call that helper, and by then the randomness has already happened inside the model's own initialization.

Note that the seed fixes the RNG stream, not the library that produces it. `check_transformers_version` pins transformers exactly, but `nn.init` belongs to torch, which is pinned nowhere, so a torch upgrade can still change the weights.

## Adding a new tiny model

1. Pick the right subfolder based on the model class suffix (`ForCausalLM`, `ForSequenceClassification`, `ForConditionalGeneration`).
2. Copy an existing script with the closest shape and adapt it — reference model id, config class, model class, special kwargs.
3. If the model requires a release newer than the TRL floor, pass it explicitly: `check_transformers_version("X.Y.Z")`. Otherwise leave the call argument-less to default to the floor.
4. Keep the `set_seed()` call that the copied script already has, directly after the version check.
5. Run it. Inspect the `[smoke_test]` and `[config_diff]` output before letting it push.
