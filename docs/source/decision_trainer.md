# DecisionTrainer

`DecisionTrainer` trains decision models using their supervised loss. Transformers owns the architecture, checkpoint format, and objective; TRL handles optimization, distributed training, and metrics. The model returns a mean `loss` and candidate `logits`. Model-specific token layouts and batching live in example collators. The library provides `DataCollatorForDecision` for shared request preparation and target normalization.

## Usage

Pass a configured model and a collator matching its forward inputs:

```python
from trl.experimental.decision import DecisionConfig, DecisionTrainer

trainer = DecisionTrainer(
    model=model,
    args=DecisionConfig(output_dir="decision-output"),
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=collator,
)
trainer.train()
trainer.save_model()
trainer.end()
```

The trainer preserves which parameters are trainable. Freeze the backbone or prepare adapters before constructing it. Alternatively, pass `peft_config` to wrap the whole model through PEFT; configure `modules_to_save` for any head that should remain trainable. Resume with the same model and adapter setup. The Jeva example saves a native base checkpoint before wrapping it with PEFT, so the adapter can be reloaded with its decision head.

## Jeva example

The [BANKING77 example](https://github.com/huggingface/trl/tree/main/examples/jeva_banking77) provides Jeva's model construction and a 1,000-step LoRA configuration with a stratified validation split:

```bash
python examples/jeva_banking77/jeva_banking77.py --config examples/jeva_banking77/config.yaml
```

> [!TIP]
> This example currently requires [Transformers #50614](https://github.com/huggingface/transformers/pull/50614). CLM and JEV use their own Transformers PRs linked below.

Jeva's `--mode` selects head, LoRA, or full training. Its token limits, truncation policy, and option chunk size are example arguments, rather than trainer settings.

Requests use Transformers' `state` / `questions` schema, with one question per training row. The example stores requests as JSON strings to preserve candidate names and order in Arrow datasets. Targets are separate:

| Field | Meaning |
|---|---|
| `target_index` | Zero-based option index, or `None` for a nullable choice. |
| `target_probs` | Null probability, then one probability per option. |
| `p_true` | Probability of true for a `noul` question. |
| `score` | Numeric score, interpolated between rubric levels. |

Choice questions use a `criteria` mapping and may set `allow_null=True`. Score questions use a `criteria` list and
may supply numeric `values`. The [Jeva example collator](https://github.com/huggingface/trl/blob/main/examples/jeva_banking77/collator.py)
uses Transformers' template to prepare separate state and option spans. It subclasses `DataCollatorForDecision`,
which delegates request rendering to `model.decision_config.prepare_inputs`. Preprocess once with
`dataset.map(collator.tokenize, remove_columns=dataset.column_names)` so batches only need padding.


## CLM example

The [typed-decisions example](https://github.com/huggingface/trl/tree/main/examples/clm_typed_decisions) trains
CLM projection heads on a frozen SmolLM2-135M encoder. It requires [Transformers #50649](https://github.com/huggingface/transformers/pull/50649).

```bash
python examples/clm_typed_decisions/clm_typed_decisions.py --config examples/clm_typed_decisions/config.yaml
```

The [CLM example collator](https://github.com/huggingface/trl/blob/main/examples/clm_typed_decisions/collator.py) reuses the request schema above and packs each batch into one prompt with a pool of distinct
candidates. Soft targets supervise a bidirectional contrastive loss. Validation holds out source-row IDs before
expanding their questions. `eval_accuracy` uses each question's own candidates; contrastive negatives are local
to each device's batch. Context parallelism is unsupported for this example.

## JEV

JEV uses [Transformers #50586](https://github.com/huggingface/transformers/pull/50586). Use the
[JEV example collator](https://github.com/huggingface/trl/blob/main/examples/jev_typed_decisions/collator.py) with the model's tokenizer and `decision_config`, and set both `per_device_train_batch_size` and
`per_device_eval_batch_size` to `1`. Disable gradient checkpointing for the current JEV implementation. Each row contains one question; use accumulation for more decisions per update.
The collator returns packed option targets for JEV's native KL and ordinal RPS loss. Predictions are padded per-question
probabilities, like the other collators. Nullable decisions are unsupported.

## Labels and metrics

Collators may return integer class labels or soft target distributions. Pass `option_mask` when padding candidate sets. With `allow_null`, target distributions contain null first and the model may return `coverage_logits`. Nullable decisions require a coverage output. `is_choice` can exclude binary and score rows from choice accuracy.

The trainer logs candidate loss, choice and candidate accuracy, Brier score, and entropy. `loss` and `eval_loss` are the native training objective; the reported candidate cross-entropy can differ from it. Null and coverage metrics are logged when present; auxiliary loss is logged only when returned by the model. Metrics aggregate across ranks and accumulation steps, and evaluation removes padded examples. `compute_metrics` receives predicted probabilities and the original labels, with `-100` marking padded candidate columns.

## Distributed training

DDP, DeepSpeed ZeRO, and FSDP2 use the usual Trainer configuration. FSDP2 currently requires Accelerate main,
included in the example script dependencies. Context and sequence parallelism are unsupported: the current model
readouts need complete sequences. Tensor and masked expert parallelism use Transformers' `distributed_config`
when loading a complete checkpoint. Token-dispatch expert parallelism is not supported by Trainer.

The trainer normalizes the native mean loss by decision count across ranks and accumulation steps. It does not
add MoE router regularization; auxiliary losses must come from the model itself. CLM's contrastive negatives remain
local to each microbatch.

## Inference

Use the model's Transformers inference API. For the LoRA example, load the saved native `base_model` and its adapter with PEFT, then call `merge_and_unload()` before creating the decision pipeline. Saved Jeva checkpoints work with the decision pipeline:

```python
from transformers import AutoModelForDecision, AutoTokenizer, pipeline

model = AutoModelForDecision.from_pretrained("decision-output")
tokenizer = AutoTokenizer.from_pretrained("decision-output")
decide = pipeline("decision", model=model, tokenizer=tokenizer)
answers = decide(request)
```

## DecisionConfig

[[autodoc]] experimental.decision.DecisionConfig

## DecisionTrainer

[[autodoc]] experimental.decision.DecisionTrainer

## DataCollatorForDecision

[[autodoc]] experimental.decision.DataCollatorForDecision
    - prepare_example
