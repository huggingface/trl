# Experimental Features

The `trl.experimental` namespace provides a minimal, clearly separated space for fast iteration on new ideas.

<Tip warning={true}>

**Stability contract:** Anything under `trl.experimental` may change or be removed in *any* release (including patch versions) without prior deprecation. Do not rely on these APIs for production workloads.

</Tip>

## Current Experimental Features

The following modules are currently available under [`trl.experimental`](https://github.com/huggingface/trl/tree/main/trl/experimental).
This list is not exhaustive and may change at any time.

### BEMA for Reference Model

This feature implements the BEMA algorithm to update the reference model during DPO training.

```python
from trl.experimental.bema_for_ref_model import BEMACallback, DPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


pref_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

bema_callback = BEMACallback(update_ref_model=True)

model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
tokenizer.pad_token = tokenizer.eos_token

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
    callbacks=[bema_callback],
)

trainer.train()
```

### GFPO

This feature implements the GFPO algorithm to enforce concise reasoning in the model's output generation, as proposed in the paper [Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning](https://huggingface.co/papers/2508.09726).

To activate GFPO in [`GFPOTrainer`]:

- set `num_remains_in_group` in [`GFPOConfig`]
- define a group filter function and set it to `group_filter_func` in [`GFPOTrainer`]. `group_filter_func` will score the `num_generations` completions and The GFPOTrainer filters groups according to their scores to get top `num_remains_in_group` completions as a new group. Model will be trained on the filtered group.

```python
# train_gfpo.py
from trl.experimental.gfpo import GFPOConfig, GFPOTrainer

# dummy group filter to scores the completions based on its indice in group
class GroupFilter:
    def __call__(self, group_completions, group_rewards, **kwargs):
        group_scores = []
        for completions, rewards in zip(group_completions, group_rewards):
            scores = [float(i) for i in range(len(completions))]
            group_scores.append(scores)
        return group_scores

training_args = GFPOConfig(
    output_dir="Qwen3-0.6B-GFPO",
    per_device_train_batch_size=4,
    num_remains_in_group=2,
    bf16=True,
)
trainer = GFPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
    group_filter_func=GroupFilter(),
)
trainer.train()
```

### GSPO-token

In the paper [Group Sequence Policy Optimization](https://huggingface.co/papers/2507.18071), the authors propose a token-level objective variant to GSPO, called GSPO-token. To use GSPO-token, you can use the `GRPOTrainer` class in `trl.experimental.gspo_token`.

```python
from trl.experimental.gspo_token import GRPOTrainer
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence_token",
    ...
)
```

<Tip warning={true}>

To leverage GSPO-token, the user will need to provide the per-token advantage  \\( \hat{A_{i,t}} \\) for each token  \\( t \\) in the sequence  \\( i \\) (i.e., make  \\( \hat{A_{i,t}} \\) varies with  \\( t \\)—which isn't the case here,  \\( \hat{A_{i,t}}=\hat{A_{i}} \\)). Otherwise, GSPO-Token gradient is just equivalent to the original GSPO implementation.

</Tip>

### GRPO With Replay Buffer

This experimental trainer, trains a model with GRPO but replaces groups (and corresponding completions) that have 0 standard deviation with groups with high rewards and standard deviation that've been used to train a model in prior batches.

#### Usage

```python
from trl.experimental.grpo_with_replay_buffer import GRPOWithReplayBufferTrainer
from datasets import load_dataset

dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

# Guarantee that some rewards have 0 std
def custom_reward_func(completions, **kwargs):
    if torch.rand(1).item() < 0.25:
        return [0] * len(completions)  # simulate some None rewards
    else:
        return torch.rand(len(completions)).tolist()

training_args = GRPOWithReplayBufferConfig(
    output_dir=self.tmp_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_generations=4,
    max_completion_length=8,
    replay_buffer_size=8,
    report_to="none",
)
trainer = GRPOTrainer(
    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    reward_funcs=[custom_reward_func],
    args=training_args,
    train_dataset=dataset,
)

previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

trainer.train()
```

To silence the runtime notice:

```bash
export TRL_EXPERIMENTAL_SILENCE=1
```

## Promotion Path (Simple)

1. **Prototype outside the main repo:** Start development in your own fork or a separate repository to iterate quickly.
2. **Experimental inclusion:** Once it’s ready for early users, move the idea into `trl.experimental.<feature>`.
3. **Improve:** Add tests, a short doc/example, and demonstrate the usage.
4. **Promote:** Once the API proves stable and there is clear interest or adoption from the community, move it into `trl.<feature>` (stable module).

## FAQ

**Why not just use branches?**
Because branches are not shipped to users; experimental code inside the package lets early adopters try things and give feedback.

**Can these APIs change or vanish without warning?**
Yes. Anything inside `trl.experimental` can change or disappear in *any* release.

**Should I use this in production?**
Only if you are fine with updating your code quickly when things change.

**Will maintainers promptly fix issues in `trl.experimental`?**
Not necessarily. The experimental module is a playground for new ideas, and maintainers may not prioritize bug fixes or feature requests there. Issues may remain unresolved until (or unless) the feature graduates to the stable API.
