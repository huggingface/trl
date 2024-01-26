# pip install openrlbenchmark==0.2.1a5
# see https://github.com/openrlbenchmark/openrlbenchmark#get-started for documentation
echo "we deal with $TAGS_STRING"

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=trl_ppo_trainer_config.value.reward_model&cen=trl_ppo_trainer_config.value.exp_name&metrics=env/reward_mean&metrics=objective/kl' \
        "ppo$TAGS_STRING" \
    --env-ids sentiment-analysis:lvwerra/distilbert-imdb \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/$FOLDER_STRING/ppo \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=output_dir&cen=_name_or_path&metrics=train/rewards/accuracies&metrics=train/loss' \
        "gpt2$TAGS_STRING" \
    --env-ids dpo_anthropic_hh \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/$FOLDER_STRING/dpo \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=output_dir&cen=_name_or_path&metrics=train/loss&metrics=eval/accuracy&metrics=eval/loss' \
        "facebook/opt-350m$TAGS_STRING" \
    --env-ids reward_modeling_anthropic_hh \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/$FOLDER_STRING/reward_modeling \
    --scan-history

python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=_step&ceik=output_dir&cen=_name_or_path&metrics=train/loss' \
        "facebook/opt-350m$TAGS_STRING" \
    --env-ids sft_openassistant-guanaco \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --output-filename benchmark/trl/$FOLDER_STRING/sft \
    --scan-history

python benchmark/upload_benchmark.py \
    --folder_path="benchmark/trl/$FOLDER_STRING" \
    --path_in_repo="images/benchmark/$FOLDER_STRING" \
    --repo_id="trl-internal-testing/example-images" \
    --repo_type="dataset"

