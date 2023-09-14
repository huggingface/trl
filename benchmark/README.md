# TRL Benchmark

This is a benchmark for TRL. Here we show the command to run it in a slurm cluster, but it can be easily adapted to run locally.

There are several benchmark axes we want to explore:

- w/ different models (gpt2, gpt2-xl, falcon, llama2)
    - key research engineering questions
        - how do different model sizes scale?
        - **given that the preference labels come from a source model `M_s` (e.g., gpt2), how does that affect the performance of a target model `M_t` (e.g., falcon, gptj, llama2)?**
            - This is actually an important assumption we have been operating.
- w/ and w/o gradient accumulation / multi-GPU
    - key research engineering question: do we need to whiten advantage across the entire batch?
- w/ and w/o peft
    - key research engineering question: how well does PEFT work with RL
- w/ and w/o quantization or 4 bits
    - key research engineering question: how well does quantization work with RL training
- w/ and w/o deepspeed
    - sanity check to make sure it works.
- w/ different datasets
    - TRL’s typical imdb sentiment
    - OAI’s sentiment dataset (https://github.com/openai/lm-human-preferences)
    - summarize from feedback ( https://github.com/openai/summarize-from-feedback)
    - helpfulness vs harmlessness (https://huggingface.co/datasets/Anthropic/hh-rlhf)


## Benchmark commands


```bash
export WANDB_ENTITY=huggingface
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.log_with wandb" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
```

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/benchmark/v0.4.7-55-g110e672/sentiment.png)



## w/ and w/o gradient accumulation
```bash
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_step_grad_accu --ppo_config.mini_batch_size 1 --ppo_config.gradient_accumulation_steps 128 --ppo_config.log_with wandb" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
```

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/benchmark/v0.4.7-55-g110e672/gradient_accu.png)


## w/ different models (gpt2, gpt2-xl, falcon, llama2)

```bash
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_gpt2 --ppo_config.log_with wandb" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_gpt2xl_grad_accu --ppo_config.model_name gpt2-xl --ppo_config.mini_batch_size 16 --ppo_config.gradient_accumulation_steps 8 --ppo_config.log_with wandb" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_falcon_rw_1b --ppo_config.model_name tiiuae/falcon-rw-1b --ppo_config.log_with wandb" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
```

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/benchmark/v0.4.7-55-g110e672/different_models.png)

## w/ and w/o PEFT
```
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_peft --use_peft --ppo_config.log_with wandb" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
```

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/benchmark/v0.4.7-55-g110e672/peft.png)
