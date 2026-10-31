# DecisionTrainer

`DecisionTrainer` trains models that return a supervised decision loss. [Jeva](https://huggingface.co/papers/2610.18773) is the first supported model: it predicts candidate probabilities and whether any candidate applies, without producing vocabulary logits. Transformers owns the model and loss; TRL handles data, metrics, and distributed training.

> [!TIP]
> This trainer currently requires [Transformers #50203](https://github.com/huggingface/transformers/pull/50203).

```bash
pip install 'git+https://github.com/huggingface/transformers.git@refs/pull/50203/head'
```

## Usage

```python
import json
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, JevaConfig, JevaForDecision
from trl.experimental.decision import DataCollatorForJeva, DecisionConfig, DecisionTrainer

encoder = AutoModel.from_pretrained("Qwen/Qwen3-0.6B")
model = JevaForDecision(JevaConfig(text_config=encoder.config, projection_dim=128), model=encoder)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
request = {
    "state": "My card was charged twice.",
    "questions": {"intent": {
        "type": "choice", "instructions": "Which intent?",
        "criteria": {"refund": None, "delivery": None}, "allow_null": True,
    }},
}
dataset = Dataset.from_list([{"request": json.dumps(request), "target_index": 0}])
trainer = DecisionTrainer(
    model=model,
    args=DecisionConfig(output_dir="jeva-output"),
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorForJeva(tokenizer, model.decision_config),
)
trainer.train()
trainer.save_model()
trainer.end()
```

`mode="head"` trains the decision head, `"lora"` also trains encoder adapters, and `"full"` trains the whole model. Pass a PEFT configuration with `mode="lora"`, or load a checkpoint with an existing adapter. Set `option_chunk_size` to limit options per encoder forward.

The [BANKING77 example](https://github.com/huggingface/trl/tree/main/examples/jeva_banking77) includes a 1,000-step LoRA configuration, a stratified validation split, and evaluation metrics.

## Dataset format

Requests use Transformers' `state` / `questions` schema. Each training row contains one question. Store the request as a JSON string in a `request` column to preserve candidate names and order in Arrow datasets. The collator also accepts separate `state` and `questions` fields.

Targets are separate from the request. Supply exactly one:

| Field | Meaning |
|---|---|
| `target_index` | Zero-based option index, or `None` for a nullable choice. |
| `target_probs` | Probability of null, then one probability per option. |
| `p_true` | Probability of true for a `noul` question. |
| `score` | Numeric score, interpolated between rubric levels. |

Choice questions use a `criteria` mapping and may set `allow_null=True`. Noul questions use `type="noul"`. Score questions use a `criteria` list and may supply strictly increasing numeric `values`. The model's decision config determines option order. `target_probs` must sum to one, with zero null mass unless null is enabled.

`DataCollatorForJeva` uses the model's decision template. It right-pads states and options separately and rejects inputs exceeding its token limits unless `truncation="truncate"` is set.

## Inference

Saved checkpoints load through Transformers:

```python
from transformers import AutoModelForDecision, AutoTokenizer, pipeline

model = AutoModelForDecision.from_pretrained("jeva-output")
tokenizer = AutoTokenizer.from_pretrained("jeva-output")
decide = pipeline("decision", model=model, tokenizer=tokenizer)
answers = decide(request)
```

Nullable choices return `null_probability` and can return `choice=None`. Direct forward outputs expose conditional candidate log probabilities as `logits` and coverage logits as `coverage_logits`.

## Metrics and distributed training

The trainer logs candidate and coverage losses, choice and candidate accuracy, null precision and recall, null rate, Brier scores, and entropy. Candidate accuracy considers decisions with a hard choice target that is present. MoE models also log their auxiliary loss. Metrics aggregate across ranks and accumulation steps; evaluation removes padded examples.

DDP, DeepSpeed ZeRO, and FSDP2 use the usual Trainer configuration. Use Accelerate main for FSDP2 until the [FP32 output fix](https://github.com/huggingface/accelerate/pull/4403) is released. Torch context parallelism requires FSDP2, SDPA, and a full causal attention encoder; pooled representations are reduced across context ranks. Recurrent encoders and DeepSpeed Ulysses are unsupported. Tensor and masked expert parallelism use Transformers' `distributed_config` when loading a complete saved checkpoint. Token-dispatch expert parallelism is not supported by Trainer.

## DecisionConfig

[[autodoc]] experimental.decision.DecisionConfig

## DecisionTrainer

[[autodoc]] experimental.decision.DecisionTrainer

## DataCollatorForJeva

[[autodoc]] experimental.decision.DataCollatorForJeva
