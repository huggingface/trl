# DecisionTrainer

Use `DecisionTrainer` to fine-tune a decision model on labeled choices, yes/no questions, or scores.

## Usage

`DataCollatorForDecision` renders each request with `model.decision_config.prepare_inputs()` and normalizes its
target. It passes the rendered prompts and targets to the model's `prepare_inputs_for_decision()` hook, which
builds the tensors for the forward pass, including labels.

For a model that provides the hook, such as JEV, pass it to the collator:

```python
from trl.experimental.decision import DataCollatorForDecision, DecisionConfig, DecisionTrainer

collator = DataCollatorForDecision(
    tokenizer,
    model.decision_config,
    prepare_inputs_for_decision=model.prepare_inputs_for_decision,
)

trainer = DecisionTrainer(
    model=model,
    args=DecisionConfig(
        output_dir="decision-output",
        per_device_train_batch_size=1,
        activation_checkpointing=False,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=collator,
)
trainer.train()
trainer.save_model()
trainer.end()
```

The Jeva and CLM scripts use the same collator and pass model-specific options, such as span lengths, to their hooks.

The trainer preserves which parameters are trainable. Freeze the backbone or prepare adapters before constructing it. Alternatively, pass `peft_config` to wrap the whole model through PEFT; configure `modules_to_save` for any head that should remain trainable. Resume with the same model and adapter setup. The Jeva example saves a native base checkpoint before wrapping it with PEFT, so the adapter can be reloaded with its decision head.

## Dataset format

Requests use Transformers' `state` / `questions` schema, with one question per training row. Store the request as a
JSON string to preserve candidate names and order in Arrow datasets. Each row also supplies one target field.

| Field | Meaning |
|---|---|
| `target_index` | Zero-based option index, or `None` for a nullable choice. |
| `target_probs` | A mapping from option names to probabilities, or a list with the null probability first followed by the option probabilities in order. |
| `p_true` | Probability of true for a `noul` question. |
| `score` | Numeric score, interpolated between rubric levels. |

With named probabilities, omitted options receive zero mass. Use the list form to assign probability to null.
For Arrow datasets with different option names across rows, store this column as `datasets.Json()`.

Choice questions use a `criteria` mapping and can set `allow_null=True` with a model that supports it. Score questions
use a `criteria` list and can supply numeric `values` for the rubric.

## Jeva example

The [BANKING77 script](https://github.com/huggingface/trl/blob/main/examples/decisions/jeva_banking77.py) supports head,
LoRA, and full training with a stratified validation split. It requires [Transformers #50614](https://github.com/huggingface/transformers/pull/50614).

```bash
python examples/decisions/jeva_banking77.py --config examples/decisions/jeva_banking77.yaml
```

Jeva's hook batches the states and options separately. The script exposes span lengths, truncation, and option chunk size.

## CLM example

The [typed-decisions script](https://github.com/huggingface/trl/blob/main/examples/decisions/clm_typed_decisions.py) trains
CLM projection heads on a frozen SmolLM2-135M encoder. It requires [Transformers #50649](https://github.com/huggingface/transformers/pull/50649).

```bash
python examples/decisions/clm_typed_decisions.py --config examples/decisions/clm_typed_decisions.yaml
```

CLM's hook packs each batch into one prompt with a pool of distinct candidates. Soft targets supervise a bidirectional
contrastive loss. Validation holds out source-row IDs before
expanding their questions. `eval_accuracy` uses each question's own candidates; contrastive negatives are local
to each device's batch. Context parallelism is unsupported for this example.

## JEV

The [JEV training script](https://github.com/huggingface/trl/blob/main/examples/decisions/jev.py) uses the shared
collator with the hook from [Transformers #51382](https://github.com/huggingface/transformers/pull/51382).
Use batch size one for training and evaluation, with one question per row. Disable activation checkpointing and
use gradient accumulation for more decisions per update.

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
