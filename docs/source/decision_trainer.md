# DecisionTrainer

`DecisionTrainer` trains decision models using their supervised loss. Transformers owns the architecture, checkpoint format, and objective; TRL handles optimization, distributed training, and metrics. The model returns a mean `loss` and candidate `logits`. Tokenization and batching belong to the model's collator.

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

The trainer preserves which parameters are trainable. Freeze the backbone or prepare adapters before constructing it. Alternatively, pass `peft_config` to wrap the whole model through PEFT; configure `modules_to_save` for any head that should remain trainable. Resume with the same model and adapter setup.

## Jeva example

The [BANKING77 example](https://github.com/huggingface/trl/tree/main/examples/jeva_banking77) provides Jeva's model construction and a 1,000-step LoRA configuration with a stratified validation split:

```bash
python examples/jeva_banking77/jeva_banking77.py --config examples/jeva_banking77/config.yaml
```

> [!TIP]
> This example currently requires [Transformers #50203](https://github.com/huggingface/transformers/pull/50203). Other decision heads need supervised-loss support before they can train through this API.

Jeva's `--mode` selects head, LoRA, or full training. Its token limits, truncation policy, and option chunk size are example arguments, rather than trainer settings.

Requests use Transformers' `state` / `questions` schema, with one question per training row. The example stores requests as JSON strings to preserve candidate names and order in Arrow datasets. Targets are separate:

| Field | Meaning |
|---|---|
| `target_index` | Zero-based option index, or `None` for a nullable choice. |
| `target_probs` | Null probability, then one probability per option. |
| `p_true` | Probability of true for a `noul` question. |
| `score` | Numeric score, interpolated between rubric levels. |

Choice questions use a `criteria` mapping and may set `allow_null=True`. Score questions use a `criteria` list and may supply numeric `values`. The library's `DataCollatorForJeva` renders the model's decision template, splits state and option spans, and converts targets to tensors. Import it from `trl.experimental.decision` and pass `tokenizer` and `model.decision_config`. Models using the same input layout can reuse it; other layouts need their own collator.

## Labels and metrics

Collators may return integer class labels or soft target distributions. Pass `option_mask` when padding candidate sets. With `allow_null`, target distributions contain null first and the model may return `coverage_logits`. Nullable decisions require a coverage output. `is_choice` can exclude binary and score rows from choice accuracy.

The trainer logs candidate loss, choice and candidate accuracy, Brier score, and entropy. Null and coverage metrics are logged when present; MoE auxiliary loss is logged when returned by the model. Metrics aggregate across ranks and accumulation steps, and evaluation removes padded examples. `compute_metrics` receives predicted probabilities and the original labels, with `-100` marking padded candidate columns.

## Distributed training

DDP, DeepSpeed ZeRO, and FSDP2 use the usual Trainer configuration. Use Accelerate main for FSDP2 until the [FP32 output fix](https://github.com/huggingface/accelerate/pull/4403) is released. Context parallelism also requires a model readout that supports sharded sequences; the Jeva example is tested with full causal encoders, FSDP2, and SDPA. Recurrent encoders and DeepSpeed Ulysses are unsupported.

Tensor and masked expert parallelism use Transformers' `distributed_config` when loading a complete checkpoint. Token-dispatch expert parallelism is not supported by Trainer. Models compute their own auxiliary loss; the trainer normalizes the total mean loss by the number of decisions across ranks and accumulation steps.

## Inference

Use the model's Transformers inference API. Saved Jeva checkpoints work with the decision pipeline:

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

## DataCollatorForJeva

[[autodoc]] experimental.decision.DataCollatorForJeva
