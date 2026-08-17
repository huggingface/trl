#!/usr/bin/env bash
# AsyncGRPO checkpoint round-trip on HF Jobs: one job trains a few steps and pushes its checkpoint to the Hub, a
# second job pulls that checkpoint back and continues from it — same trackio run, same prompt-stream position.
#
#   ./hfjob_async_grpo_smoke.sh fresh    # steps 1..4, pushes <HUB_MODEL_ID>/last-checkpoint
#   ./hfjob_async_grpo_smoke.sh resume   # resumes that checkpoint, steps 5..8
#   ./hfjob_async_grpo_smoke.sh logs     # follow the most recent of the two
#
# The script submits *itself* as the job's command (`bash -c "$(cat "$0")" … in-job`), so there is nothing to upload
# and no second file to keep in sync; the `in-job` stage below is what actually runs inside the container.
#
# Runs inside `vllm/vllm-openai:latest` on an h200x2 job. Two GPUs are required, not a convenience: the
# weight-transfer NCCL group has world_size = vllm_world_size + 1, and two ranks cannot share one device. FA3
# (`kernels-community/flash-attn3`, hardcoded by the trainer) is why the flavor is Hopper.
set -euo pipefail

BRANCH=${BRANCH:-trl-checkpoint}
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
HUB_MODEL_ID=${HUB_MODEL_ID:-aminediroHF/async-grpo-ckpt-smoke-r1d-1.5b}
RUN_NAME=${RUN_NAME:-async-grpo-ckpt-smoke}
# Completions are capped at 2048 tokens; 1024 of prompt headroom on top of that is plenty for this dataset.
MAX_MODEL_LEN=3072

STAGE=${1:-}

submit() {
    local stage=$1 max_steps=$2 resume=$3 job_id
    # `-q` reduces stdout to the job id so it can be captured; `--` keeps the inlined script below out of reach of the
    # CLI's global flag stripper. --timeout covers image pull + vLLM load + the steps + a ~9 GB checkpoint upload.
    job_id=$(uvx hf jobs run \
        -q \
        --flavor h200x2 \
        --timeout 50m \
        --detach \
        --name "async-grpo-ckpt-${stage}" \
        --secrets HF_TOKEN \
        -e "SMOKE_STAGE=${stage}" \
        -e "BRANCH=${BRANCH}" \
        -e "MAX_STEPS=${max_steps}" \
        -e "SAVE_STEPS=4" \
        -e "RESUME_FROM_HUB=${resume}" \
        -e "HUB_MODEL_ID=${HUB_MODEL_ID}" \
        -e "RUN_NAME=${RUN_NAME}" \
        -- \
        vllm/vllm-openai:latest \
        bash -c "$(cat "$0")" async-grpo-ckpt-smoke in-job)
    echo "$job_id" > "/tmp/async-grpo-ckpt-${stage}.job"
    echo "submitted ${stage} (max_steps=${max_steps}, resume=${resume}): ${job_id}"
    echo "follow with: $0 logs"
}

in_job() {
    export HF_HOME=/tmp/hf_cache
    export PYTHONUNBUFFERED=1
    # trl warns on every import from trl.experimental; silence it to keep the log readable.
    export TRL_EXPERIMENTAL_SILENCE=1

    echo "=== [1/5] deps (stage=${SMOKE_STAGE}, branch=${BRANCH}, max_steps=${MAX_STEPS}) ==="
    # The image ships python3.12 with torch 2.13/cu130, vllm 0.27.1 and transformers 5.15 — new enough for both the
    # nccl weight-transfer backend and the trainer, so nothing in that stack gets upgraded here. There is no `python`
    # and no `git` in this image: use `python3`, and fetch the branch as a tarball.
    python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
    curl -sL "https://github.com/huggingface/trl/archive/refs/heads/${BRANCH}.tar.gz" | tar xz -C /tmp
    TRL_DIR=/tmp/trl-${BRANCH}
    # `kernels` is what fetches `kernels-community/flash-attn3` at the first forward (hardcoded by the trainer).
    pip install -q "$TRL_DIR" kernels trackio math-verify latex2sympy2_extended
    python3 -c "import trl; print('trl', trl.__version__, 'from', trl.__file__)"

    echo "=== [2/5] vLLM on GPU 1 ==="
    # No --model-impl: weights sync into vLLM's native implementation. --logprobs-mode processed_logprobs is
    # load-bearing — `old_log_probs` (the PPO denominator) comes from these logprobs. --dtype bfloat16 matches the
    # trainer's `dtype="bfloat16"`, so the precision-mismatch warning at train begin should stay silent.
    CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
        --host 0.0.0.0 --port 8000 \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.6 \
        --logprobs-mode processed_logprobs \
        --weight-transfer-config '{"backend":"nccl"}' > /tmp/vllm.log 2>&1 &
    VLLM_PID=$!

    echo "=== [3/5] wait for /health (up to 20 min) ==="
    for i in $(seq 1 240); do
        if curl -sf http://localhost:8000/health > /dev/null; then echo "vLLM ready after ${i}x5s"; break; fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM died:"; tail -50 /tmp/vllm.log; exit 1; fi
        sleep 5
    done
    curl -sf http://localhost:8000/health > /dev/null || { echo "vLLM never became healthy:"; tail -50 /tmp/vllm.log; exit 1; }

    # Direct check of what `VLLMClient.get_dtype()` parses, before the trainer relies on it.
    echo "--- /server_info dtype ---"
    curl -s 'http://localhost:8000/server_info?config_format=json' \
      | python3 -c "import json,sys; print('served dtype:', json.load(sys.stdin)['vllm_config']['model_config']['dtype'])"

    echo "=== [4/5] training on GPU 0 ==="
    # `fresh` ends with "[done] global_step=4, rows_consumed=N" and a checkpoint at <HUB_MODEL_ID>/last-checkpoint.
    # `resume` starts with "[resume] …: global_step=4, rows_consumed=N", logs "Resuming the prompt stream at dataset
    # row N", and ends at global_step=8 without having replayed the first prompts.
    cd "$TRL_DIR"
    CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/async_grpo_checkpoint_smoke.py

    echo "=== [5/5] done; vLLM tail ==="
    tail -20 /tmp/vllm.log
    kill "$VLLM_PID" 2>/dev/null || true
}

case "$STAGE" in
    fresh)  submit fresh 4 0 ;;
    resume) submit resume 8 1 ;;
    logs)
        id_file=$(ls -t /tmp/async-grpo-ckpt-*.job 2>/dev/null | head -1 || true)
        [ -n "$id_file" ] || { echo "no job id saved yet; submit 'fresh' or 'resume' first" >&2; exit 1; }
        uvx hf jobs logs --follow "$(cat "$id_file")"
        ;;
    in-job) in_job ;;
    *)      echo "usage: $0 {fresh|resume|logs}" >&2; exit 2 ;;
esac
