# PPO recipe: SFT + Reward Model + PPO on Pythia-1B-deduped + TL;DR

This directory ships an example PPO training script ([`ppo_tldr.py`](./ppo_tldr.py)) that fine-tunes [`EleutherAI/pythia-1b-deduped`](https://huggingface.co/EleutherAI/pythia-1b-deduped) on the [TL;DR summarization task](https://huggingface.co/datasets/trl-lib/tldr) using a reward model. That script assumes you already have an **SFT checkpoint** and a **reward model checkpoint**. This README documents the commands that produce those two checkpoints from the base model with TRL's own [`SFTTrainer`](../../../trl/trainer/sft_trainer.py) and [`RewardTrainer`](../../../trl/trainer/reward_trainer.py), so the full SFT → reward modeling → PPO pipeline is reproducible end-to-end. See [issue #2015](https://github.com/huggingface/trl/issues/2015) for background.

## Quickstart: use the published checkpoints

If you do not want to train the SFT and reward models yourself, the TRL team has published a ready-to-use pair on the Hub:

- SFT: [`trl-lib/pythia-1b-deduped-tldr-sft`](https://huggingface.co/trl-lib/pythia-1b-deduped-tldr-sft)
- Reward model: [`trl-lib/pythia-1b-deduped-tldr-rm`](https://huggingface.co/trl-lib/pythia-1b-deduped-tldr-rm)

These are already used by [`examples/scripts/xpo.py`](../xpo.py) and [`examples/scripts/nash_md.py`](../nash_md.py). To plug them into [`ppo_tldr.py`](./ppo_tldr.py), pass them as `--sft_model_path` and `--reward_model_path`. The rest of this README walks through producing equivalent checkpoints from scratch.

## Verify your setup first (optional)

Before committing many GPU-hours to the full recipe, two cheap checks:

- **Pipeline smoke test** (~3–5 minutes on Apple Silicon / single laptop GPU; uses tiny CI fixtures and 2 optimizer steps per stage) — confirms the SFT → RM → PPO command chain runs end-to-end without crashing:
  ```bash
  bash examples/scripts/ppo/smoke_test.sh
  ```
- **Reward-model diagnostics on the published checkpoint** — load `trl-lib/pythia-1b-deduped-tldr-rm` and inspect its real chosen/rejected accuracy, margin distribution, length bias, and failure cases. See the walkthrough in [`recipe.ipynb`](./recipe.ipynb) and the script at [`evaluate_reward_model.py`](./evaluate_reward_model.py).

### What real numbers look like

On a recent run of `bash smoke_test.sh` (Apple Silicon M-series, ~4 minutes total): all three stages exited 0; SFT printed `train_loss 0.675` over 2 steps; PPO produced 4 episodes of value/policy loss updates.

On a 200-pair sample of `trl-lib/tldr-preference` (validation split) scored by the published `trl-lib/pythia-1b-deduped-tldr-rm` (Apple Silicon, MPS, ~15 minutes):

| Metric                 | Value  | Interpretation                                                  |
|------------------------|--------|-----------------------------------------------------------------|
| `accuracy`             | 0.595  | The RM prefers `chosen` over `rejected` in 59.5% of pairs — only modestly above the 0.5 chance baseline. |
| `mean_margin`          | +0.647 | On average chosen scores higher than rejected, by 0.65 reward units. |
| `mispreferred_fraction`| 0.405  | 40.5% of pairs have margin ≤ 0 — the RM gets them backwards.    |
| `near_zero_fraction`   | 0.26   | An additional 26% sit within ±0.5 of zero — low-confidence cases. |
| `length_bias_pearson`  | +0.158 | Modest positive correlation between `len(chosen) − len(rejected)` and margin — some length bias, but not the dominant failure mode at this scale. |

A reward model that's only 60% accurate provides a weak training signal for PPO, which is consistent with the instability reported in [issue #2015](https://github.com/huggingface/trl/issues/2015). The diagnostics above were produced by [`evaluate_reward_model.py`](./evaluate_reward_model.py); run it against your own RM checkpoint and compare. Note that small samples (n ≤ 50) are noisy enough to be misleading on these statistics — prefer n ≥ 200.

## Step 1 — Train the SFT model

The SFT step fine-tunes the base Pythia model on the prompt/completion pairs in [`trl-lib/tldr`](https://huggingface.co/datasets/trl-lib/tldr) using [`trl/scripts/sft.py`](../../../trl/scripts/sft.py). Hyperparameters mirror the `sft.py` docstring example (1 epoch, packing, batch size 2 × 8-step accumulation, learning rate 2.0e-5); `max_length` is left at its default (1024).

Single GPU:

```bash
python trl/scripts/sft.py \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --dataset_name trl-lib/tldr \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 25 \
    --output_dir pythia-1b-deduped-tldr-sft
```

Multi-GPU (DeepSpeed ZeRO-2):

```bash
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    trl/scripts/sft.py \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --dataset_name trl-lib/tldr \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 25 \
    --output_dir pythia-1b-deduped-tldr-sft
```

The trained checkpoint is written to `./pythia-1b-deduped-tldr-sft`. Pass `--push_to_hub` to publish it.

## Step 2 — Train the reward model

The reward modeling step trains a sequence-classification head on top of the SFT checkpoint from Step 1, using the chosen/rejected pairs in [`trl-lib/tldr-preference`](https://huggingface.co/datasets/trl-lib/tldr-preference) and [`examples/scripts/reward_modeling.py`](../reward_modeling.py). Hyperparameters mirror the `reward_modeling.py` docstring example (1 epoch, batch size 8, learning rate 1.0e-5); `max_length` is left at its default (1024).

Single GPU:

```bash
python examples/scripts/reward_modeling.py \
    --model_name_or_path pythia-1b-deduped-tldr-sft \
    --dataset_name trl-lib/tldr-preference \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 25 \
    --output_dir pythia-1b-deduped-tldr-rm
```

Multi-GPU (DeepSpeed ZeRO-2):

```bash
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/reward_modeling.py \
    --model_name_or_path pythia-1b-deduped-tldr-sft \
    --dataset_name trl-lib/tldr-preference \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 25 \
    --output_dir pythia-1b-deduped-tldr-rm
```

At the end of training, `reward_modeling.py` calls `trainer.evaluate()` and logs the chosen-vs-rejected accuracy. Before moving on to PPO, confirm this accuracy is meaningfully above chance (>50%); a reward model at chance will not provide useful PPO signal.

For a more thorough diagnostic — margin distribution, length-bias correlation, near-zero-margin bucket, and a CSV dump of failure cases — run [`evaluate_reward_model.py`](./evaluate_reward_model.py) against your checkpoint:

```bash
python examples/scripts/ppo/evaluate_reward_model.py \
    --reward_model_path pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr-preference \
    --split validation \
    --num_samples 1000 \
    --output_dir reward_model_eval
```

Writes `reward_model_eval/summary.json` and `reward_model_eval/failures.csv`.

## Step 3 — Run PPO with your checkpoints

Plug the two output directories from Steps 1 and 2 into [`ppo_tldr.py`](./ppo_tldr.py):

```bash
python examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --output_dir pythia-1b-deduped-tldr-ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path pythia-1b-deduped-tldr-sft \
    --reward_model_path pythia-1b-deduped-tldr-rm \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --eval_strategy steps \
    --eval_steps 100
```

This is the same command as the existing [`ppo_tldr.py`](./ppo_tldr.py) docstring, with `--sft_model_path` and `--reward_model_path` pointed at your locally trained checkpoints. A DeepSpeed ZeRO-2 multi-GPU variant is shown in [`ppo_tldr.py`](./ppo_tldr.py)'s own docstring.

## Notes

- **Tokenizer consistency.** Step 2 (`reward_modeling.py`) loads its base tokenizer from your Step 1 output. With current TRL trainers this produces SFT and RM checkpoints whose saved `tokenizer_config.json` and `chat_template.jinja` are identical — empirically verified by `smoke_test.sh` (diff between the two configs is a single load-state field). This is the inconsistency [issue #2015](https://github.com/huggingface/trl/issues/2015) reports between the `cleanrl/EleutherAI_pythia-1b-deduped__{sft,reward}__tldr` checkpoints (where the RM's saved `pad_token` is `None`); the same metadata quirk also affects the published `trl-lib/pythia-1b-deduped-tldr-rm`, which predates current save behavior. **Re-training with this recipe produces a clean pair.**
- **Why this recipe does not bit-reproduce the `cleanrl/*` checkpoints.** The `cleanrl/EleutherAI_pythia-1b-deduped__{sft,reward}__tldr` checkpoints (currently the default `--sft_model_path` / `--reward_model_path` in [`ppo_tldr.py`](./ppo_tldr.py)'s docstring) were produced by a different codebase with different training scripts and tokenizer setup, which is the source of the pad-token / chat-template inconsistency reported in [issue #2015](https://github.com/huggingface/trl/issues/2015). This recipe instead targets the maintained TRL pair `trl-lib/pythia-1b-deduped-tldr-{sft,rm}` (used today by [`xpo.py`](../xpo.py) and [`nash_md.py`](../nash_md.py)) — produced with TRL trainers and reproducible from TRL. To use the trl-lib pair with `ppo_tldr.py`, just pass them explicitly as `--sft_model_path trl-lib/pythia-1b-deduped-tldr-sft --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm`.
- **Reproducibility caveat.** The exact hyperparameters used to produce `trl-lib/pythia-1b-deduped-tldr-{sft,rm}` are not separately published. The values above are taken from the docstring examples of [`trl/scripts/sft.py`](../../../trl/scripts/sft.py) and [`examples/scripts/reward_modeling.py`](../reward_modeling.py) and are known to produce reasonable checkpoints; tune them for your compute budget.
- **Same recipe, different downstream trainer.** The same SFT + reward model pair also feeds the RLOO, XPO, and Nash-MD examples ([`examples/scripts/rloo.py`](../rloo.py), [`examples/scripts/xpo.py`](../xpo.py), [`examples/scripts/nash_md.py`](../nash_md.py)).
