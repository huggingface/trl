# pip install openrlbenchmark==0.2.1a5
# see https://github.com/openrlbenchmark/openrlbenchmark#get-started for documentation
echo "we deal with $TAGS_STRING"

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=trl_ppo_trainer_config.value.reward_model&cen=trl_ppo_trainer_config.value.exp_name&metrics=env/reward_mean&metrics=objective/kl' \
        "ppo$TAGS_STRING" \
        "ppo_gpt2xl_grad_accu$TAGS_STRING" \
    --env-ids sentiment-analysis:lvwerra/distilbert-imdb \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/$FOLDER_STRING/different_models \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=trl_ppo_trainer_config.value.reward_model&cen=trl_ppo_trainer_config.value.exp_name&metrics=env/reward_mean&metrics=objective/kl' \
        "ppo_Cerebras-GPT-6.7B_grad_accu_deepspeed_stage2$TAGS_STRING" \
    --env-ids sentiment-analysis:cerebras/Cerebras-GPT-6.7B \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/$FOLDER_STRING/deepspeed \
    --scan-history

python benchmark/upload_benchmark.py \
    --folder_path="benchmark/trl/$FOLDER_STRING" \
    --path_in_repo="images/benchmark/$FOLDER_STRING" \
    --repo_id="trl-internal-testing/example-images" \
    --repo_type="dataset"

