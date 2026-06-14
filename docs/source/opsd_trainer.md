# OPSD

On-Policy Self-Distillation (OPSD) was introduced in [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://huggingface.co/papers/2601.18734) by [Siyan Zhao](https://huggingface.co/siyanzhao), Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover.

> Large language models are increasingly post-trained on reasoning tasks, yet prevailing paradigms face inherent trade-offs: supervised fine-tuning on expert demonstrations suffers from distribution mismatch with the student's own generations, while reinforcement learning provides only sparse outcome-level feedback. We propose On-Policy Self-Distillation (OPSD), where a single model serves as both student and teacher through differential conditioning contexts: the student observes only the problem, while the teacher additionally receives the ground-truth solution. Matching their token-level distributions along on-policy student trajectories yields dense supervision without an external teacher, and a per-token pointwise KL clipping mechanism prevents high-divergence stylistic tokens from dominating the training signal.

## How it works

For each problem the student samples its own completion. The teacher, which is the same network conditioned on the problem plus the ground-truth solution from the `privileged_context` column (wrapped in the `teacher_prompt_template`), then scores that completion token by token, and the divergence between the two next-token distributions is minimized along the student trajectory. Gradients flow only through the student logits; the teacher acts as a fixed supervision target.

## Loss modes and clipping

`distillation_mode` selects the objective: `"full_logits"` (the default and the paper's setting) computes the full-vocabulary divergence, `"topk_logits"` restricts it to the teacher's top-`distillation_topk` support, and `"sampled_token"` uses a token-level reverse KL on the realized tokens (requires `distillation_alpha=1.0`). `distillation_alpha` interpolates the divergence between forward KL (`0.0`, the official setting), reverse KL (`1.0`), and the generalized JSD in between.

`distillation_kl_clip` (default `0.05`) applies the paper's pointwise per-vocabulary-entry clipping: each entry's divergence contribution is capped before the vocabulary sum, which keeps high-divergence style tokens such as reasoning connectives from dominating the training signal. Set it to `None` to disable. Setting `use_liger_kernel=True` swaps in a memory-efficient fused JSD loss (Liger) for `full_logits`; the fused kernel cannot express the pointwise clip, so it requires `distillation_kl_clip=None` and is incompatible with `distillation_is_clip`.

## Choosing the teacher

`teacher_model_kind` selects the teacher weights: `"base"` (the default and the official OPSD setting) freezes the initial student, `"live"` reuses the current student, and `"ema"` tracks the student with an exponential moving average synced every `teacher_sync_steps` steps at rate `teacher_update_rate`. With PEFT, the `base` teacher is realized by disabling the adapter, so no second model copy is needed.

The teacher and student can use different chat-template settings via `teacher_chat_template_kwargs`, for example pairing a non-thinking student with a thinking teacher on Qwen3 models (the configuration the paper found strongest): `chat_template_kwargs={"enable_thinking": False}` with `teacher_chat_template_kwargs={"enable_thinking": True}`.

## Usage

```python
from datasets import Dataset

from trl.experimental.opsd import OPSDConfig, OPSDTrainer

dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": "Solve 2+2."}]],
        "privileged_context": ["The answer is 4."],
    }
)

training_args = OPSDConfig(
    output_dir="opsd-model",
    max_completion_length=1024,
)

trainer = OPSDTrainer(
    model="Qwen/Qwen3-1.7B",
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

## Expected dataset columns

Each example must provide:

- `prompt`: the student-facing problem
- `privileged_context`: the ground-truth solution, shown only to the teacher (the same contract as the SDFT and SDPO trainers)

## Serving the teacher from the vLLM server

With `teacher_model_kind="live"` the teacher is the current student, whose weights the vLLM **server** already holds (they are synced for generation each step). Set `use_teacher_server=True` to score the teacher log-probabilities on that same server instead of running a separate local teacher forward, removing the teacher from the training step entirely:

```python
training_args = OPSDConfig(
    output_dir="opsd-model",
    use_vllm=True,
    vllm_mode="server",
    teacher_model_kind="live",
    use_teacher_server=True,
    distillation_mode="sampled_token",
    distillation_alpha=1.0,
    distillation_kl_clip=None,
)
```

When using the teacher server:

- `use_vllm=True` and `vllm_mode="server"` are required
- `teacher_model_kind` must be `"live"` (the server holds the current student weights), which deviates from the paper's `"base"` teacher
- `distillation_mode` must be `"sampled_token"` (reverse KL on the realized token) or `"topk_logits"`. The server returns the teacher's own top-k log-probs. `full_logits` is unavailable, so `distillation_kl_clip` must be `None` for `sampled_token`.
- `use_liger_kernel` is not supported

## Callbacks

The trainer emits a small set of callback hooks that are useful for debugging, observability, and tests:

- `on_self_distillation_batch_prepared`: fired when a self-distillation batch is ready. The payload includes `prompt_ids`, `completion_ids`, `teacher_input_ids`, and `old_per_token_logps` when importance-sampling clipping inputs are available.
- `on_generation_batch_built`: fired when a new buffered generation batch is created. The payload includes `generate_every` and `steps_per_generation`.

## Example script

Use [`trl/experimental/opsd/opsd.py`](https://github.com/huggingface/trl/blob/main/trl/experimental/opsd/opsd.py) to launch OPSD training from the command line. It maps gsm8k-style `question`/`answer` or `problem`/`solution` columns automatically (use `--solution_column` for other names) and supports PEFT/LoRA via the standard `ModelConfig` flags.

```bash
accelerate launch trl/experimental/opsd/opsd.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name open-thoughts/OpenThoughts-114k \
    --output_dir opsd-qwen3-1.7b \
    --learning_rate 5e-6 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --dtype bfloat16 \
    --bf16 true
```

## OPSDConfig

[[autodoc]] experimental.opsd.OPSDConfig

## OPSDTrainer

[[autodoc]] experimental.opsd.OPSDTrainer
    - train
    - save_model
    - push_to_hub
