export WANDB_ENTITY=huggingface
export WANDB_PROJECT=trl
bash $BENCHMARK_SCRIPT > output.txt

# Extract Job IDs into an array
job_ids=($(grep "Job ID:" output.txt | awk '{print $3}'))

# Extract WANDB_TAGS into an array
WANDB_TAGS=($(grep "WANDB_TAGS:" output.txt | awk '{print $2}'))
WANDB_TAGS=($(echo $WANDB_TAGS | tr "," "\n"))

# Print to verify
echo "Job IDs: ${job_ids[@]}"
echo "WANDB_TAGS: ${WANDB_TAGS[@]}"

TAGS_STRING="?tag=${WANDB_TAGS[0]}"
FOLDER_STRING="${WANDB_TAGS[0]}"
for tag in "${WANDB_TAGS[@]:1}"; do
    TAGS_STRING+="&tag=$tag"
    FOLDER_STRING+="_$tag"
done

echo "TAGS_STRING: $TAGS_STRING"
echo "FOLDER_STRING: $FOLDER_STRING"

TAGS_STRING=$TAGS_STRING FOLDER_STRING=$FOLDER_STRING BENCHMARK_PLOT_SCRIPT=$BENCHMARK_PLOT_SCRIPT sbatch --dependency=afterany:$job_ids benchmark/post_github_comment.sbatch
