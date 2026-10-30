# Jeva

TRL supports [Jeva: System One Models from First Principles](https://huggingface.co/papers/2610.18773) by Qi Li and
Xinchao Wang. A shared encoder represents the state and each option independently. A small decision stack predicts
candidate probabilities and a separate coverage estimate for decisions where none of the options apply.

## Usage

```python
from datasets import Dataset
from transformers import AutoTokenizer
from trl.experimental.jeva import JevaConfig, JevaModel, JevaTrainer

backbone = "Qwen/Qwen3-0.6B"
model = JevaModel.from_backbone_pretrained(backbone, dtype="float32")
tokenizer = AutoTokenizer.from_pretrained(backbone)
dataset = Dataset.from_list([
    {
        "state": "I was charged twice.",
        "question": "Which team should handle this request?",
        "options": ["Billing: payments and refunds", "Engineering: software failures"],
        "target_index": 0,
    },
])
trainer = JevaTrainer(
    model=model,
    processing_class=tokenizer,
    args=JevaConfig(output_dir="jeva-output", mode="head"),
    train_dataset=dataset,
)
trainer.train()
trainer.save_model()
```

For a complete training script, see the [BANKING77 example](https://github.com/huggingface/trl/tree/main/examples/jeva_banking77).

`mode="head"` trains the projections and decision stack while keeping the encoder frozen. Use `mode="full"` to
train everything, or `mode="lora"` with an existing adapter or `peft_config=LoraConfig(task_type="FEATURE_EXTRACTION",
...)`. The encoder uses `AutoModel`, so training does not construct language-model vocabulary logits.

To import `config.yaml`/`head.pt` checkpoints, use `JevaModel.from_jeva_checkpoint("LIQIIIII/Jeva-4B")` and pass the
checkpoint's tokenizer to the trainer. Trainer outputs use the Transformers format: reload them with
`JevaModel.from_pretrained("jeva-output")`, or resume with `trainer.train(resume_from_checkpoint=...)`.

## Dataset format

Rows contain `state`, `question`, and `options`. `decision_type` accepts `"choice"` (the default), `"noul"`, and
`"score"`. Supply one label field:

| Label | Meaning |
| :--- | :--- |
| `target_index` | Zero-based candidate index, or `None` for null when `allow_null=True`. |
| `target_probs` | Soft distribution with null first, then one probability per option. |
| `p_true` | Noul truth probability; options default to `["false", "true"]`. |
| `score` | Score interpolated between neighboring ordered rubric `values`. |

Only Choice supports null. Score values default to option indices; Noul can specify custom options and a
`positive_index`. Use `target_probs` as a common label column when mixing task types.

The loss combines candidate cross entropy with coverage cross entropy for null-enabled rows. Each decision has
equal weight, regardless of its option count. Predictions and targets have null in the first column; evaluation
pads missing candidate columns with `-100`, which should be masked in `compute_metrics`.

## Usage tips

- Set `max_state_length` and `max_option_length` separately. Overflow raises unless `truncation="truncate"`.
- Lower `option_chunk_size` to encode fewer options per forward. It keeps all candidates in the decision.
- Use your usual Accelerate configuration for DDP, FSDP2, or DeepSpeed ZeRO. The trainer keeps each decision's options
  together and aligns encoder chunk counts across ranks. Use process-based training instead of DataParallel.
- MoE load balancing uses the encoder's `router_aux_loss_coef`; set it to zero to disable it. It covers state and
  option tokens and is disabled for a frozen encoder.

For expert or tensor parallelism, save a JevaModel first, then load the complete checkpoint with
`JevaModel.from_pretrained(..., distributed_config=...)`. Use masked expert plans (`ep_router` and `moe_tp_experts`)
with `ep_size=tp_size`. Token-dispatch EP is rejected until Transformers Trainer supports it. Follow Transformers'
checkpoint restrictions for models sharded at load time.

Torch context parallelism uses FSDP2, `cp_size > 1`, BF16, SDPA, and Transformers 5.22.1 or newer.
The trainer shards state and option sequences and reduces pooled representations across context ranks. Evaluation
supports `eval_loss` only; prediction gathering and `compute_metrics` are unavailable. This works
with full causal attention encoders; recurrent, sliding-window, and chunked-attention encoders are excluded.
DeepSpeed Ulysses is not supported. Standard Transformers model loading also honors registered component patches
and attention/kernel implementations.

## JevaTrainer

[[autodoc]] experimental.jeva.JevaTrainer
    - train
    - save_model
    - push_to_hub

## JevaConfig

[[autodoc]] experimental.jeva.JevaConfig

## JevaModel

[[autodoc]] experimental.jeva.JevaModel

## JevaModelConfig

[[autodoc]] experimental.jeva.JevaModelConfig
