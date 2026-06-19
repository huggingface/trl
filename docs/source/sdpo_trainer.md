# SDPO

Self-Distillation Policy Optimization (SDPO) was introduced in [Reinforcement Learning via Self-Distillation](https://huggingface.co/papers/2601.20802) by [Jonas Hübotter](https://huggingface.co/jonhue), Frederike Lübeck, Lejs Behric, [Anton Baumann](https://huggingface.co/antonbaumann), Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause.

> Large language models are increasingly post-trained with reinforcement learning in verifiable domains such as code and math. Yet, current methods for reinforcement learning with verifiable rewards (RLVR) learn only from a scalar outcome reward per attempt, creating a severe credit-assignment bottleneck. Many verifiable environments actually provide rich textual feedback, such as runtime errors or judge evaluations, that explain why an attempt failed. We formalize this setting as reinforcement learning with rich feedback and introduce Self-Distillation Policy Optimization (SDPO), which converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model. SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed next-token predictions back into the policy. In this way, SDPO leverages the model's ability to retrospectively identify its own mistakes in-context. Across scientific reasoning, tool use, and competitive programming on LiveCodeBench v6, SDPO improves sample efficiency and final accuracy over strong RLVR baselines. Notably, SDPO also outperforms baselines in standard RLVR environments that only return scalar feedback by using successful rollouts as implicit feedback for failed attempts. Finally, applying SDPO to individual questions at test time accelerates discovery on difficult binary-reward tasks, achieving the same discovery probability as best-of-k sampling or multi-turn conversations with 3x fewer attempts.

## How it works

SDPO targets reinforcement learning with verifiable rewards (RLVR), where each attempt yields only a sparse scalar reward. It turns that into a dense, token-level signal: for each prompt the policy samples `num_generations` completions scored by `reward_funcs`, a successful rollout (plus optional `privileged_context` feedback) becomes a teacher reprompt, and the teacher's feedback-informed distribution over a completion is distilled back into the policy. Teacher and student are the same network, so no external teacher or reward model is needed beyond the verifier.

## Loss modes and the teacher

`distillation_weight` controls how the two signals combine as a convex combination: the loss is `(1 - distillation_weight) * policy_loss + distillation_weight * distillation_loss`. `1.0` (the default) trains purely on the self-distillation loss, `0.0` falls back to the standard GRPO-style policy gradient, and intermediate values blend both. The distillation objective itself is set by `distillation_mode` — `"sampled_token"` (the default) uses a token-level reverse KL and requires `distillation_alpha=1.0`, while `"full_logits"` and `"topk_logits"` distill over the full or top-`distillation_topk` vocabulary. Setting `use_liger_kernel=True` swaps in a memory-efficient fused JSD loss (Liger) for the distillation term; it requires `distillation_weight=1.0`, `distillation_mode="full_logits"`, and is incompatible with `distillation_is_clip`.

`teacher_model_kind` chooses the teacher weights: `"ema"` (the default) tracks the student with an exponential moving average synced every `teacher_sync_steps` steps at rate `teacher_update_rate`, `"live"` reuses the current student directly, and `"base"` freezes the initial weights. Reprompting is governed by `use_successful_as_teacher`, `success_reward_threshold`, `dont_reprompt_on_self_success`, and the `reprompt_template` / `solution_template` / `feedback_template` strings. Generation runs through transformers by default, or vLLM (colocate or server mode) when `use_vllm=True`.

## Expected dataset columns

Each example must provide:

- `prompt`: the student-facing prompt
- `privileged_context`: optional privileged text, such as environment feedback, used when `include_environment_feedback=True`

## Usage

```python
from datasets import Dataset

from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": "Solve 2+2."}]],
        "privileged_context": ["Your earlier answer used the wrong format."],
    }
)

training_args = SDPOConfig(
    output_dir="sdpo-model",
    distillation_mode="topk_logits",       # Explicitly select top-K logit distillation
    distillation_topk=100,                 # Required when using top-K logit distillation
    include_environment_feedback=True,     # Use dataset privileged_context for teacher reprompts
)

trainer = SDPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

SDPO always requires a `prompt` column. To use environment feedback, also include a `privileged_context` column and set `include_environment_feedback=True`. SDPO will use successful rollouts and, when enabled, that text to build teacher reprompts for self-distillation.

## Serving the teacher from the vLLM server

With `teacher_model_kind="live"` the teacher is the current student, whose weights the vLLM **server** already holds (they are synced for generation each step). Set `use_teacher_server=True` to score the teacher log-probabilities on that same server instead of running a separate local teacher forward, removing the teacher from the training step entirely:

```python
training_args = SDPOConfig(
    output_dir="sdpo-model",
    use_vllm=True,
    vllm_mode="server",
    teacher_model_kind="live",
    use_teacher_server=True,
    distillation_weight=1.0,
    distillation_mode="sampled_token",
)
```

When using the teacher server:

- `use_vllm=True` and `vllm_mode="server"` are required
- `teacher_model_kind` must be `"live"` (the server holds the current student weights)
- `distillation_weight` must be `1.0` (pure distillation; a convex blend with the policy loss needs the full-vocabulary logits)
- `distillation_mode` must be `"sampled_token"` (reverse KL on the realized token) or `"topk_logits"`. The server returns the teacher's own top-k log-probs, so `topk_logits` distills over the teacher's top-k support (it cannot use the student's, unlike the local objective); with a `"live"` teacher the two supports nearly coincide. `full_logits` is unavailable.
- `use_liger_kernel` is not supported

## Callbacks

The trainer emits a small set of callback hooks that are useful for debugging, observability, and tests. These hooks are intended as practical integration points for experimental self-distillation workflows.

Shared self-distillation hooks:

- `on_self_distillation_batch_prepared`: fired when a self-distillation batch is ready. The payload includes `prompt_ids`, `completion_ids`, and `old_per_token_logps` when importance-sampling clipping inputs are available.
- `on_generation_batch_built`: fired when a new buffered generation batch is created. The payload includes `generate_every` and `steps_per_generation`.

SDPO-specific hook:

- `on_teacher_context_built`: fired after SDPO constructs the teacher-conditioned inputs. The payload includes `teacher_input_ids`, `teacher_attention_mask`, `completion_mask`, and `self_distillation_mask`.

## Example script

Use [`trl/experimental/sdpo/sdpo.py`](https://github.com/huggingface/trl/blob/main/trl/experimental/sdpo/sdpo.py) to launch SDPO training from the command line. The script supports verifiable math rewards, environment feedback via `--feedback_column`, and PEFT/LoRA via the standard `ModelConfig` flags.

```bash
python trl/experimental/sdpo/sdpo.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --dataset_name openai/gsm8k \
    --dataset_config main \
    --output_dir outputs/sdpo-qwen35-2b-gsm8k \
    --learning_rate 5e-5 \
    --dtype bfloat16 \
    --bf16 true \
    --max_completion_length 128 \
    --use_peft \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --generation_batch_size 32 \
    --distillation_alpha 1.0 \
    --distillation_mode sampled_token \
    --distillation_weight 0.5 \
    --report_to none \
    --eval_strategy steps \
    --eval_steps 1000 \
    --save_strategy no
```

## SDPOConfig

[[autodoc]] experimental.sdpo.SDPOConfig

## SDPOTrainer

[[autodoc]] experimental.sdpo.SDPOTrainer
    - train
    - save_model
    - push_to_hub
