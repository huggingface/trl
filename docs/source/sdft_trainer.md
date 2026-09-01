# SDFT

Self-Distilled Fine-Tuning (SDFT) is described in the paper [Self-Distillation Enables Continual Learning](https://huggingface.co/papers/2601.19897) by Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal.

> Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce Self-Distillation Fine-Tuning (SDFT), a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.

## How it works

Plain supervised fine-tuning trains on the demonstration text off-policy, which tends to overwrite prior capabilities. SDFT learns on-policy instead: the student generates from the plain `prompt`, a teacher — the same model shown the `prompt` plus the example's `privileged_context` — re-scores those tokens, and its demonstration-conditioned distribution is distilled back into the student. Teacher and student are one network differing only in what they see, creating a *self*-distillation loop.

## Choosing the teacher

`teacher_model_kind` selects which copy of the model acts as teacher. `"base"` (the default) freezes the initial weights as a fixed reference, matching the paper; `"live"` reuses the current student for a zero-lag self-teacher; `"ema"` maintains an exponential moving average, resynced every `teacher_sync_steps` steps at rate `teacher_update_rate`. Under PEFT, `"base"` is obtained by disabling the adapter during the teacher forward to recover the base weights, and `"ema"` with pure-LoRA training holds the moving average in a dedicated `"teacher"` adapter instead of a second model copy. `"ema"` with a non-pure-LoRA PEFT model (e.g. `modules_to_save` or `bias`) is not supported, since a separate EMA copy cannot be parameter-matched to the student.

By default the student generates from the plain prompt; set `generate_from_teacher=True` to sample from the demonstration-conditioned prompt instead, trading on-policy fidelity for higher-quality rollouts. The distillation objective is set by `distillation_mode` (`"topk_logits"` by default, with `"full_logits"`, `"sampled_token"`, and `"dopd"` alternatives), `distillation_alpha`, and `distillation_topk`; `num_loss_tokens_to_skip` drops leading completion tokens from the loss. Setting `use_liger_kernel=True` swaps in a memory-efficient fused JSD loss (Liger) that avoids materializing the full-vocabulary logits; it requires `distillation_mode="full_logits"` and is incompatible with `distillation_is_clip`. Training is text-only; generation runs through transformers by default, or vLLM (colocate or server mode) when `use_vllm=True`.

## Usage

```python
from datasets import Dataset

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": "Solve 2+2."}]],
        "privileged_context": ["Example answer: 4."],
    }
)

training_args = SDFTConfig(
    output_dir="sdft-model",
    distillation_alpha=0.5,
    distillation_mode="topk_logits",
    distillation_topk=5,
    max_completion_length=64,
)

trainer = SDFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

To generate from the teacher-conditioned prompt instead of the student prompt, set `generate_from_teacher=True`.
To customize how the teacher prompt is built, set `teacher_prompt_template` on [`SDFTConfig`].

## DOPD: advantage-gap token routing

Set `distillation_mode="dopd"` to enable DOPD ([Dual On-Policy Distillation](https://huggingface.co/papers/2606.30626)) instead of a single objective applied uniformly to every token. DOPD routes each completion token into one of four regimes based on the "advantage gap" — the absolute log-probability difference on the realized token between the privileged teacher and the privileged student (the student's own forward on the same privileged context, used only for routing) — and each side's confidence: low-gap-confident tokens get a light top-k reverse-KL nudge toward the teacher, low-gap-unsure tokens get a weak stop-gradient self-regularization anchor, high-gap-teacher-confident tokens get the full corrective JSD signal, and high-gap-student-confident tokens get a light stop-gradient consistency nudge. `teacher_model_kind="live"` is not supported with `distillation_mode="dopd"`, since the routing needs a genuinely different privileged-student anchor.

```python
training_args = SDFTConfig(
    distillation_mode="dopd",
    distillation_dopd_gap_threshold=2.0,               # nats separating "low gap" / "high gap" regimes
    distillation_dopd_confidence_threshold=0.5,        # max-softmax-prob threshold for "confident"
    distillation_dopd_light_topk=8,                    # top-k support for the light-signal regimes
    distillation_dopd_self_reg_weight=0.01,            # weight on the low-gap/unsure self-regularization regime
    distillation_dopd_student_consistency_weight=0.1,  # weight on the high-gap/student-confident regime
)
```

## Serving the teacher from the vLLM server

With `teacher_model_kind="live"` the teacher is the current student, whose weights the vLLM **server** already holds (they are synced for generation each step). Set `use_teacher_server=True` to score the teacher log-probabilities on that same server instead of running a separate local teacher forward, removing the teacher from the training step entirely:

```python
training_args = SDFTConfig(
    output_dir="sdft-model",
    use_vllm=True,
    vllm_mode="server",
    teacher_model_kind="live",
    use_teacher_server=True,
    distillation_mode="sampled_token",
)
```

When using the teacher server:

- `use_vllm=True` and `vllm_mode="server"` are required
- `teacher_model_kind` must be `"live"` (the server holds the current student weights)
- `distillation_mode` must be `"sampled_token"` (reverse KL on the realized token) or `"topk_logits"`. The server returns the teacher's own top-k log-probs, so `topk_logits` distills over the teacher's top-k support (it cannot use the student's, unlike the local objective); with a `"live"` teacher the two supports nearly coincide. `full_logits` is unavailable. `dopd` is also unavailable: it needs full student/teacher logit access to compute per-token confidence and the advantage gap, the same reason `full_logits` is excluded.
- `use_liger_kernel` is not supported

## Expected dataset columns

Each example must provide:

- `prompt`: the student-facing prompt
- `privileged_context`: only the extra teacher-only information, such as a demonstration, hint, or privileged feedback

Both standard text prompts and conversational prompts are supported by the trainer prompt handling.

## Callbacks

The trainer emits a small set of callback hooks that are useful for debugging, observability, and tests. These hooks are intended as practical integration points for experimental self-distillation workflows.

Shared self-distillation hooks:

- `on_self_distillation_batch_prepared`: fired when a self-distillation batch is ready. The payload includes `prompt_ids`, `completion_ids`, and `old_per_token_logps` when importance-sampling clipping inputs are available.
- `on_generation_batch_built`: fired when a new buffered generation batch is created. The payload includes `generate_every` and `steps_per_generation`.

SDFT-specific hook:

- `on_generation_prompts_selected`: fired when SDFT chooses the prompt source for on-policy generation. The payload includes the selected `generation_prompts` and the corresponding `generation_prompt_text`.

## Example script

Use [`examples/sdft_privileged_context/sdft_privileged_context.py`](https://github.com/huggingface/trl/blob/main/examples/sdft_privileged_context/sdft_privileged_context.py) to launch SDFT training from the command line. The script supports any causal LM from the Hub, custom local datasets via `--dataset_path`, and PEFT/LoRA via the standard `ModelConfig` flags.

```bash
python examples/sdft_privileged_context/sdft_privileged_context.py \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --dataset_name your-org/your-dataset \
    --output_dir outputs/sdft-qwen3.5-0.8b \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --generate_from_teacher \
    --teacher_model_kind ema \
    --teacher_sync_steps 1 \
    --teacher_update_rate 0.05 \
    --eval_strategy steps \
    --eval_steps 50 \
    --report_to wandb
```

The original implementation is available at [idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation).

## SDFTConfig

[[autodoc]] experimental.sdft.SDFTConfig

## SDFTTrainer

[[autodoc]] experimental.sdft.SDFTTrainer
    - train
    - save_model
    - push_to_hub
