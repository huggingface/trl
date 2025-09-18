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

### GRPO With Replay Buffer
This experimental trainer, trains a model with GRPO but replaces groups (and corresponding completions) that have 0 standard deviation with groups with high rewards and standard deviation that've been used to train a model in prior batches.

## Usage

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
