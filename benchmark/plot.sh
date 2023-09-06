# pip install openrlbenchmark==0.2.1a5
PR_662_TAG=v0.4.7-55-g110e672
PR_662_NAME=PR-662

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=trl_ppo_trainer_config.value.reward_model&cen=trl_ppo_trainer_config.value.exp_name&metrics=env/reward_mean&metrics=objective/kl' \
        "sentiment_tuning?tag=$PR_662_TAG&cl=sentiment lvwerra/gpt2-imdb ($PR_662_NAME)" \
    --env-ids sentiment-analysis:lvwerra/distilbert-imdb \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/sentiment \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=trl_ppo_trainer_config.value.reward_model&cen=trl_ppo_trainer_config.value.exp_name&metrics=env/reward_mean&metrics=objective/kl' \
        "sentiment_tuning?tag=$PR_662_TAG&cl=sentiment lvwerra/gpt2-imdb ($PR_662_NAME)" \
        "sentiment_tuning_step_grad_accu?tag=$PR_662_TAG&cl=sentiment lvwerra/gpt2-imdb gradient accumulation ($PR_662_NAME)" \
    --env-ids sentiment-analysis:lvwerra/distilbert-imdb \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/gradient_accu \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=trl_ppo_trainer_config.value.reward_model&cen=trl_ppo_trainer_config.value.exp_name&metrics=env/reward_mean&metrics=objective/kl' \
        "sentiment_tuning?tag=$PR_662_TAG&cl=sentiment lvwerra/gpt2-imdb ($PR_662_NAME)" \
        "sentiment_tuning_gpt2?tag=$PR_662_TAG&cl=sentiment gpt2 ($PR_662_NAME)" \
        "sentiment_tuning_falcon_rw_1b?tag=$PR_662_TAG&cl=sentiment tiiuae/falcon-rw-1b ($PR_662_NAME)" \
        "sentiment_tuning_gpt2xl_grad_accu?tag=$PR_662_TAG&cl=sentiment gpt2xl ($PR_662_NAME)" \
    --env-ids sentiment-analysis:lvwerra/distilbert-imdb \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/different_models \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=trl_ppo_trainer_config.value.reward_model&cen=trl_ppo_trainer_config.value.exp_name&metrics=env/reward_mean&metrics=objective/kl' \
        "sentiment_tuning?tag=$PR_662_TAG&cl=sentiment lvwerra/gpt2-imdb ($PR_662_NAME)" \
        "sentiment_tuning_peft?tag=$PR_662_TAG&cl=sentiment lvwerra/gpt2-imdb w/ peft ($PR_662_NAME)" \
    --env-ids sentiment-analysis:lvwerra/distilbert-imdb \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/peft \
    --scan-history

python benchmark/upload_benchmark.py