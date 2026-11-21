# Decision models

These examples use Transformers' native decision models and losses with TRL's `DecisionTrainer`.
Request preparation uses the shared `DataCollatorForDecision` base and `DecisionConfig.prepare_inputs`.
Jeva and CLM define their model-specific collators inside their training scripts.

## Jeva: BANKING77

Requires [Transformers #50614](https://github.com/huggingface/transformers/pull/50614).
The supplied config trains a Jeva head and encoder LoRA on BANKING77 with a stratified validation holdout.

```bash
python examples/decisions/jeva_banking77.py --config examples/decisions/jeva_banking77.yaml
```

`--mode` selects head, LoRA, or full training. LoRA saves a native `base_model` beside the adapter for reloads.

## CLM: typed decisions

Requires [Transformers #50649](https://github.com/huggingface/transformers/pull/50649).
Trains projection heads on a frozen SmolLM2-135M encoder using `LocalLLaMA/typed-decisions`.

```bash
python examples/decisions/clm_typed_decisions.py --config examples/decisions/clm_typed_decisions.yaml
```

The collator deduplicates candidates within each batch and builds soft targets for native bidirectional InfoNCE.
Validation holds out source-row IDs before expanding their questions. Negatives remain local to each microbatch;
gradient accumulation does not enlarge the pool.

## JEV: prepared decision requests

Requires [Transformers #50586](https://github.com/huggingface/transformers/pull/50586).
Each dataset row contains one question in `request` (JSON), or `state` / `questions`, plus one of `target_index`,
`target_probs`, `p_true`, or `score`. `target_probs` contains a zero null entry followed by option probabilities.
The script prepares native model fields and uses Transformers' `default_data_collator`.

```bash
python examples/decisions/jev.py --model_name_or_path autotrust/JEV-9B \
    --dataset_name your/decision-requests --output_dir output/jev \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_checkpointing false
```

Use batch size one and accumulation for more decisions per update. The model computes KL and ordinal RPS;
nullable decisions and activation checkpointing are unsupported by the current JEV model.
