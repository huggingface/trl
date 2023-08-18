# TRL Benchmark


Benchmark axis:

- w/ different models (gpt2, gpt2-xl, falcon, llama2)
    - key research engineering questions
        - how do different model sizes scale?
        - **given that the preference labels come from a source model `M_s` (e.g., gpt2), how does that affect the performance of a target model `M_t` (e.g., falcon, gptj, llama2)?**
            - This is actually an important assumption we have been operating.
- w/ and w/o gradient accumulation
    - key research engineering question: do we need to whiten advantage across the entire batch?
- w/ and w/o peft
    - key research engineering question: how well does PEFT work with RL
- w/ and w/o FSDP
    - sanity check to make sure it works.
- w/ different datasets
    - TRL’s typical imdb sentiment
    - OAI’s sentiment dataset (https://github.com/openai/lm-human-preferences)
    - summarize from feedback ( https://github.com/openai/summarize-from-feedback)
    - helpfulness vs harmlessness (https://huggingface.co/datasets/Anthropic/hh-rlhf)


## w/ different models (gpt2, gpt2-xl, falcon, llama2)

```
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --log_with wandb" \
    --num-seeds 10 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
```
```
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --model_name gpt2-xl --log_with wandb" \
    --num-seeds 10 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
```

## w/ and w/o gradient accumulation
```
WANDB_TAGS="sentiment,grad_accu" python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --log_with wandb --mini_batch_size 1 --gradient_accumulation_steps 128" \
    --num-seeds 10 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
```