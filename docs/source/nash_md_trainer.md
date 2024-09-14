# Nash MD Trainer


This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif).

## Get started

To just run the Nash MD script to make sure this trainer can run, you can run the following command to train a Nash MD model with a dummy reward model.

```bash
python examples/scripts/nash_md.py \
    --model_name_or_path EleutherAI/pythia-14m  \
    --reward_model_path EleutherAI/pythia-14m \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-14m-tldr-nash-md \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --max_new_tokens 64 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0
```

## Explanation of the logged metrics

The logged metrics are as follows:

* `loss/score`: The mean reinforce score loss.
* `loss/kl_div`: The mean kl divergence loss.
* `objective/entropy`: The mean entropy of the model and reference data.
* `rewards/accuracies`: The accuracies of the Nash MD's implicit reward model.
* `rewards/chosen`: The mean scores (according to the reward model) of the model completions.
* `rewards/rejected`: The mean scores (according to the reward model) of the mixture completions.
* `rewards/margins`: The mean reward margin (according to reward model) between the chosen and mixture completions.
* `logps/chosen`: The mean log probabilities of the chosen completions.
* `logps/rejected`: The mean log probabilities of the reference completions.
* `val/model_contain_eos_token`: The amount of times the model's output contains the eos token.
* `val/ref_contain_eos_token`: The amount of times the mixture's output contains the eos token.

## NashMDTrainer

[[autodoc]] NashMDTrainer

## NashMDConfig

[[autodoc]] NashMDConfig
