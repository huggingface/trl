### How to check Cluster Usage / Statistics

Seeing the queue

```jsx
squeue --format="%.10i %.10P %.10u %.5t %.12M %.20l %.5D %.5q %.10R" -p hopper-prod
```

Seeing a Summary of who is using the nodes / gpu’s right now

```jsx
/admin/home/hynek_kydlicek/scripts/check_gpu_usage.sh
```

Seeing your used disk space

```jsx
du -h --max-depth=1  <path> | sort -hr
```

### How to update my jobs time limit during running

```jsx
hfctl job update JobId=12345 TimeLimit=2:00:00
```

see [hfctl](https://www.notion.so/HFCTL-Cluster-Management-Tool-2ab1384ebcac808fba8dcb248221d243?pvs=21) page for details

## Job allocations

You can run multiple sequential `srun` commands, keeping the same node reservation. It can be useful to keep access to local data (`/scratch`) between 2 runs.

```bash
# create a reservation with 1 CPU, 1G RAM
$ salloc --qos high --cpus-per-task=1 --mem=1G -t05:00:00 --no-shell
salloc: Granted job allocation 4523211
salloc: Waiting for resource configuration
salloc: Nodes ip-26-0-163-127 are ready for job
# Run a first job in this allocation
$  srun --jobid 4523211 --cpus-per-task=1 --mem=1G echo "This is a first task"
This is a first task
# Run a second job in this allocation
$ srun --jobid 4523211 --cpus-per-task=1 --mem=1G echo "This still runs in the same jobid"
This still runs in the same jobid
# /!\ Don't forget to cancel your job when done
$ scancel 4523211
```

## Interactive jobs

```bash
# Interactive session with 1 GPU:
$ srun --nodes=1 --qos=high --gres=gpu:1 \
	  --partition=hopper-prod --time 1:00:00 --pty bash

# Interactive session CPU-only:
$ srun --nodes=1 --qos=high --partition=hopper-cpu --time 1:00:00 --pty bash
```

## Heterogeneous jobs / Running TGI alongside your job

Heterogeneous jobs allow to run a single job that launches multiple processes (steps) using different resource types.

A good usage is for jobs that need TGI to generate some data: you can launch TGI and your process in the same overall job.

In the example below we launch TGI in a container with 4 GPUs, while our consuming job (a benchmark) uses 1 GPU and queries TGI.

```bash
#!/usr/bin/env bash
#SBATCH --job-name tgi-benchmark
#SBATCH --output /fsx/%u/logs/%x-%j.log
#SBATCH --time 1:50:00
#SBATCH --qos normal
#SBATCH --partition hopper-prod
#SBATCH --gpus 4 --ntasks 1 --cpus-per-task 11 --mem-per-cpu 20G --nodes=1
#SBATCH hetjob
#SBATCH --gpus 1 --ntasks 1 --cpus-per-task 11 --mem-per-cpu 20G --nodes=1

MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
echo "Starting TGI benchmark for $MODEL"
export RUST_BACKTRACE=full
export RUST_LOG=text_generation_inference_benchmark=info
export PORT=8090

echo "Model will run on ${SLURM_JOB_NODELIST_HET_GROUP_0}:${PORT}"
echo "Benchmark will run on ${SLURM_JOB_NODELIST_HET_GROUP_1}"

# start TGI
srun --het-group=0 \
     -u \
     -n 1 \
     --container-image='ghcr.io#huggingface/text-generation-inference' \
     --container-env=PORT \
     --container-mounts="/scratch:/data" \
     --container-workdir='/usr/src' \
     --no-container-mount-home \
     /usr/local/bin/text-generation-launcher \
      --model-id $MODEL \
      --max-concurrent-requests 512 &

# wait until TGI is ready to handle requests, die after 5 minutes
timeout 300 bash -c "while [[ \"\$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health)\" != \"200\" ]]; do sleep 1 && echo \"Waiting for TGI to start...\"; done" || exit 1
exit_code=$?

RESULTS_DIR="/fsx/$USER/benchmarks_results/tgi"
mkdir -p "${RESULTS_DIR}"

if [[ $exit_code != 124 ]]; then
    # run benchmark
    echo "Starting benchmark"
    srun --het-group=1 \
         -u \
         -n 1 \
         --container-image="${USER}@registry.hpc-cluster-hopper.hpc.internal.huggingface.tech#library/text-generation-inference-benchmark:latest" \
         --container-mounts="${RESULTS_DIR}:/opt/text-generation-inference-benchmark/results" \
         --no-container-mount-home \
         text-generation-inference-benchmark \
             --tokenizer-name "$MODEL" \
             --max-vus 800 \
             --url "http://${SLURM_JOB_NODELIST_HET_GROUP_0}:${PORT}" \
             --duration 30s \
             --warmup 30s \
             --num-rates 2 \
             --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
             --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
             --no-console
fi

# stop TGI
scancel --signal=TERM "$SLURM_JOB_ID+0"

echo "End of benchmark"
```

## Checking partitions state and capacity

You can add this to your `~/.bashrc` to quickly view the available nodes on a partition

```jsx
check_hopper_gpus() {
    local partition=$1

    printf "%-20s %-15s %-8s %-8s %-8s\n" "NODELIST" "GPU_STATUS" "TOTAL" "USED" "FREE"
    echo "----------------------------------------------------------------"

    sinfo -p "$partition" -N -h -O "NodeList:20,StateCompact:10,Gres:20,GresUsed:50" | awk '
    {
        # Extract total and used GPUs
        split($3, t_arr, ":"); total = t_arr[3] + 0
        split($4, u_arr, ":"); split(u_arr[3], clean_u, "("); used = clean_u[1] + 0
        free = total - used

        # Translate Slurm states into plain English based on GPU context
        raw_state = $2
        if (raw_state ~ /plnd|pow_up/) {
            status = "CLOUD_STANDBY"
        } else if (free == 0) {
            status = "NO_GPUS_LEFT"
        } else if (raw_state ~ /mix/ && free > 0) {
            status = "PARTIALLY_FREE"
        } else if (raw_state ~ /idle/) {
            status = "FULLY_FREE"
        } else if (raw_state ~ /drain|down|maint/) {
            status = "UNAVAILABLE"
        } else {
            status = raw_state  # Fallback for unexpected states
        }

        # Print formatted row (we are printing all nodes here, sorted by FREE)
        printf "%-20s %-15s %-8s %-8s %-8s\n", $1, status, total, used, free
    }' | sort -k5,5nr
}

alias check-dev='check_hopper_gpus hopper-dev'
alias check-prod='check_hopper_gpus hopper-prod'
```

[**HFCTL - Cluster Management Tool**](https://www.notion.so/HFCTL-Cluster-Management-Tool-2ab1384ebcac808fba8dcb248221d243?pvs=21)
