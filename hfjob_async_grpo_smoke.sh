#!/usr/bin/env bash
# AsyncGRPO sanity run on HF Jobs, chained across jobs by Hub checkpoints.
#
#   ./hfjob_async_grpo_smoke.sh smoke     # h200x2, 4 steps, 1 trainer + 1 vLLM GPU: exercises the whole path cheaply
#   ./hfjob_async_grpo_smoke.sh fresh     # h200x8, 4 trainer + 4 vLLM GPUs: steps 1..STEPS_PER_JOB
#   ./hfjob_async_grpo_smoke.sh resume    # same shape, continues from <HUB_MODEL_ID>/last-checkpoint
#   ./hfjob_async_grpo_smoke.sh logs      # follow the most recently submitted job
#
# `resume` is the same command every time: the in-job script reads `global_step` out of the checkpoint and trains
# STEPS_PER_JOB more, up to TOTAL_STEPS. Every job in the chain reports to one trackio run.
#
# The Slurm original (`sanity_run/async2n_job.sbatch`) had two nodes: 8 trainer ranks on one, 8 vLLM engines on the
# other. One h200x8 has to hold both, so it is split 4 + 4 and `ACCUM` doubles to keep the recipe's 128 completions per
# optimizer step (2 x 16 x 4 = 128, where the Slurm run had 2 x 8 x 8). Two GPUs is the floor, not a convenience: the
# weight-transfer NCCL group has world_size = vllm_world_size + 1, and two ranks cannot share one device. FA3
# (`kernels-community/flash-attn3`, hardcoded by the trainer) is why the flavor is Hopper.
#
# The job's code comes from a `git archive` of this branch, synced to a bucket and mounted read-only — not from
# GitHub. Fetching a branch tarball in-job failed three runs in a row with 429/503: GitHub rate-limits by IP and the
# jobs egress is shared. The branch on `origin` stays the record; the bucket is just transport.
set -euo pipefail

BRANCH=${BRANCH:-trl-checkpoint}
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# Completions are capped at the recipe's 8192 tokens; 1280 of prompt headroom on top of that is plenty for this
# dataset. Shorter caps are a false economy: at 2048 every completion is truncated, so every reward is 0 and the run
# trains on nothing.
MAX_MODEL_LEN=9472
# `bf16` and `fp16` are the paper's two arms, both mixed precision over fp32 master weights. This drives vLLM's
# `--dtype` and the trainer's mixed precision from one place so they cannot drift apart.
PRECISION=${PRECISION:-bf16}
[ "$PRECISION" = "fp16" ] && SERVE_DTYPE=float16 || SERVE_DTYPE=bfloat16

STAGE=${1:-}
SHIP_DIR=/tmp/async-grpo-ship

submit() {
    local stage=$1 flavor=$2 n_train=$3 n_vllm=$4 accum=$5 steps_per_job=$6 timeout=$7 resume=$8 out job_id
    # The tarball is built from the committed branch, so what runs is exactly what is on `origin`.
    rm -rf "$SHIP_DIR" && mkdir -p "$SHIP_DIR"
    git archive --format=tar.gz --prefix=trl-src/ "$BRANCH" > "$SHIP_DIR/trl.tar.gz"

    # `--` keeps the inlined script below out of reach of the CLI's global flag stripper, and every option has to come
    # before the image: `hf jobs run` takes IMAGE as a positional, so a flag after it is swallowed as the image name.
    out=$(uvx hf jobs run \
        --flavor "$flavor" \
        --timeout "$timeout" \
        --detach \
        --name "async-grpo-${stage}" \
        --secrets HF_TOKEN \
        -v "$SHIP_DIR:/ship" \
        -e "SMOKE_STAGE=${stage}" \
        -e "N_TRAIN=${n_train}" \
        -e "N_VLLM=${n_vllm}" \
        -e "ACCUM=${accum}" \
        -e "STEPS_PER_JOB=${steps_per_job}" \
        -e "TOTAL_STEPS=${TOTAL_STEPS}" \
        -e "SAVE_STEPS=${SAVE_STEPS}" \
        -e "RESUME_FROM_HUB=${resume}" \
        -e "HUB_MODEL_ID=${HUB_MODEL_ID}" \
        -e "RUN_NAME=${RUN_NAME}" \
        -e "PROJECT=${PROJECT}" \
        -e "PRECISION=${PRECISION}" \
        -e "SERVE_DTYPE=${SERVE_DTYPE}" \
        -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
        -e "MODEL=${MODEL}" \
        -- \
        vllm/vllm-openai:latest \
        bash -c "$(cat "$0")" async-grpo-sanity in-job)
    # The CLI formats that line differently depending on whether it thinks it is talking to a human, an agent or a
    # script, and every one of those forms carries the 24-hex job id — so pull that out rather than trusting a shape.
    job_id=$(printf '%s\n' "$out" | grep -oE '[0-9a-f]{24}' | head -1)
    echo "$job_id" > "/tmp/async-grpo-${stage}.job"
    echo "submitted ${stage} on ${flavor}: ${job_id}  (${n_train} trainer + ${n_vllm} vLLM GPUs, ${steps_per_job} steps)"
    echo "follow with: $0 logs"
}

in_job() {
    export HF_HOME=/tmp/hf_cache
    export PYTHONUNBUFFERED=1
    # trl warns on every import from trl.experimental; silence it to keep the log readable.
    export TRL_EXPERIMENTAL_SILENCE=1

    echo "=== [1/5] deps (stage=${SMOKE_STAGE}, ${N_TRAIN} trainer + ${N_VLLM} vLLM GPUs, precision=${PRECISION}) ==="
    # The image ships python3.12 with torch 2.13/cu130, vllm 0.27.1 and transformers 5.15 — new enough for both the
    # nccl weight-transfer backend and the trainer, so nothing in that stack gets upgraded here. There is no `python`
    # in this image: use `python3`.
    python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
    tar xzf /ship/trl.tar.gz -C /tmp
    TRL_DIR=/tmp/trl-src
    # `kernels` is what fetches `kernels-community/flash-attn3` at the first forward (hardcoded by the trainer).
    pip install -q "$TRL_DIR" kernels trackio math-verify latex2sympy2_extended
    python3 -c "import trl; print('trl', trl.__version__)"

    # The last N_VLLM devices serve, the first N_TRAIN train. Same node, so the weight-transfer group is local.
    VLLM_DEVICES=$(seq -s, $((N_TRAIN)) $((N_TRAIN + N_VLLM - 1)))
    TRAIN_DEVICES=$(seq -s, 0 $((N_TRAIN - 1)))

    echo "=== [2/5] vLLM on GPUs ${VLLM_DEVICES} (dp=${N_VLLM}, dtype=${SERVE_DTYPE}) ==="
    # --data-parallel-size, not tensor parallel: R1-Distill-Qwen-1.5B has 2 KV heads, so TP would replicate them and
    #   pay an all-reduce per layer on a 1.5B model. DP gives N whole engines with no cross-GPU communication.
    # --logprobs-mode processed_logprobs is load-bearing: `old_log_probs` (the PPO denominator) comes from these.
    # --generation-config vllm stops the server adopting the model's `generation_config.json` (`top_p: 0.95`, which is
    #   oat's *eval* setting) as its sampling defaults. The trainer sends top_p=1.0 explicitly, so this is belt and
    #   braces — but it is the flag whose absence cost four collapsed runs.
    CUDA_VISIBLE_DEVICES=$VLLM_DEVICES VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
        --host 0.0.0.0 --port 8000 \
        --data-parallel-size "$N_VLLM" \
        --tensor-parallel-size 1 \
        --dtype "$SERVE_DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.85 \
        --logprobs-mode processed_logprobs \
        --generation-config vllm \
        --weight-transfer-config '{"backend":"nccl"}' > /tmp/vllm.log 2>&1 &
    VLLM_PID=$!

    echo "=== [3/5] wait for /health (up to 20 min) ==="
    for i in $(seq 1 240); do
        if curl -sf http://localhost:8000/health > /dev/null; then echo "vLLM ready after ${i}x5s"; break; fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM died:"; tail -50 /tmp/vllm.log; exit 1; fi
        sleep 5
    done
    curl -sf http://localhost:8000/health > /dev/null || { echo "vLLM never became healthy:"; tail -50 /tmp/vllm.log; exit 1; }

    # A DP group that came up degraded would look healthy while generating on a fraction of the node, and the
    # weight-transfer world size would be wrong. Check the count before spending anything on training.
    WORLD=$(curl -sf --max-time 10 http://localhost:8000/get_world_size || echo "")
    echo "--- /get_world_size -> ${WORLD:-<no response>}"
    echo "$WORLD" | grep -q "\"world_size\":[[:space:]]*${N_VLLM}" || {
        echo "FATAL: expected world_size ${N_VLLM} from vLLM, got: ${WORLD:-<no response>}"; tail -50 /tmp/vllm.log; exit 1; }
    # Direct check of what `VLLMClient.get_dtype()` parses, before the trainer relies on it.
    curl -s 'http://localhost:8000/server_info?config_format=json' \
      | python3 -c "import json,sys; print('--- served dtype:', json.load(sys.stdin)['vllm_config']['model_config']['dtype'])"

    echo "=== [4/5] training on GPUs ${TRAIN_DEVICES} (accum=${ACCUM}) ==="
    # `distributed_type: MULTI_GPU` is required for multi-rank runs: under `NO`, `accelerator.device` is an indexless
    # `cuda` and the NCCL weight transfer fails with "this nccl communicator is created to work on cuda, but the input
    # tensor is on cuda:0". A single-rank run needs no launcher at all.
    cd "$TRL_DIR"
    if [ "$N_TRAIN" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES accelerate launch \
            --num_processes "$N_TRAIN" --num_machines 1 --multi_gpu --mixed_precision "$PRECISION" \
            examples/scripts/async_grpo_sanity.py
    else
        CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES python3 examples/scripts/async_grpo_sanity.py
    fi

    echo "=== [5/5] done; vLLM tail ==="
    tail -20 /tmp/vllm.log
    kill "$VLLM_PID" 2>/dev/null || true
}

case "$STAGE" in
    smoke)
        # Cheapest end-to-end rehearsal of the same code path: 1 + 1 GPUs, 8 completions per step, its own Hub repo
        # and trackio project so it cannot land next to the real curves.
        HUB_MODEL_ID=${HUB_MODEL_ID:-aminediroHF/async-grpo-ckpt-smoke-r1d-1.5b}
        RUN_NAME=${RUN_NAME:-async-grpo-ckpt-smoke}
        PROJECT=${PROJECT:-async-grpo-ckpt-smoke}
        TOTAL_STEPS=${TOTAL_STEPS:-8} SAVE_STEPS=${SAVE_STEPS:-4}
        submit smoke h200x2 1 1 8 "${STEPS_PER_JOB:-4}" "${TIMEOUT:-60m}" "${RESUME:-0}"
        ;;
    fresh|resume)
        HUB_MODEL_ID=${HUB_MODEL_ID:-aminediroHF/async-grpo-sanity-r1d-1.5b}
        RUN_NAME=${RUN_NAME:-async-grpo-hfjobs}
        PROJECT=${PROJECT:-async-grpo-sanity-r1d-1.5b}
        TOTAL_STEPS=${TOTAL_STEPS:-2400} SAVE_STEPS=${SAVE_STEPS:-150}
        [ "$STAGE" = "resume" ] && resume=1 || resume=0
        submit "$STAGE" h200x8 4 4 16 "${STEPS_PER_JOB:-300}" "${TIMEOUT:-6h}" "$resume"
        ;;
    logs)
        id_file=$(ls -t /tmp/async-grpo-*.job 2>/dev/null | head -1 || true)
        [ -n "$id_file" ] || { echo "no job id saved yet; submit 'smoke', 'fresh' or 'resume' first" >&2; exit 1; }
        uvx hf jobs logs --follow "$(cat "$id_file")"
        ;;
    in-job) in_job ;;
    *)      echo "usage: $0 {smoke|fresh|resume|logs}" >&2; exit 2 ;;
esac
