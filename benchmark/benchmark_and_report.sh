#### Step 1: create a work directory:
# this is necessary because another github action job will remove
# the entire directory, which slurm depends on.
# https://stackoverflow.com/questions/4632028/how-to-create-a-temporary-directory
MY_SLURM_TMP_DIR=/fsx/costa/slurm_tmpdir
mkdir -p $MY_SLURM_TMP_DIR
WORK_DIR=`mktemp -d -p "$MY_SLURM_TMP_DIR"`
cp -r "$PWD" "$WORK_DIR"
cd "$WORK_DIR/$(basename "$PWD")"
echo WORK_DIR: $WORK_DIR

#### Step 2: actual work starts:
echo PATH is $PATH
echo PYTHONPATH is $PYTHONPATH
echo whcih python is $(which python)

export WANDB_ENTITY=huggingface
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
