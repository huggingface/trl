#!/usr/bin/env bash
# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pipeline smoke test for the PPO TL;DR recipe.
#
# Runs SFT -> reward modeling -> PPO with the same flag shape documented in
# README.md, but swaps the production model + dataset for tiny CI fixtures
# and caps each step at 2 optimization steps. Validates that the three
# commands parse and chain together end-to-end; does NOT validate recipe
# quality (use the published trl-lib checkpoints + evaluate_reward_model.py
# for that).
#
# Runtime on Apple Silicon (MPS): ~3-5 minutes after first-time dep install.
# Uses `uv run` so dependencies are managed via each script's PEP 723 header.
#
# Usage:
#   bash examples/scripts/ppo/smoke_test.sh
#
# Outputs three throw-away checkpoint directories under /tmp/trl-smoke-* and
# prints PASS/FAIL for each stage. Leaves outputs in place on success so they
# can be inspected; cleans them up only on failure of the SFT/RM stages.

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

TINY_MODEL="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
SFT_OUT=$(mktemp -d /tmp/trl-smoke-sft-XXXXXX)
RM_OUT=$(mktemp -d /tmp/trl-smoke-rm-XXXXXX)
PPO_OUT=$(mktemp -d /tmp/trl-smoke-ppo-XXXXXX)

echo "Smoke outputs: SFT=$SFT_OUT  RM=$RM_OUT  PPO=$PPO_OUT"

echo ""
echo "=== [1/3] SFT smoke (tiny fixture, 2 steps) ==="
uv run trl/scripts/sft.py \
    --model_name_or_path "$TINY_MODEL" \
    --dataset_name trl-internal-testing/zen \
    --dataset_config standard_language_modeling \
    --learning_rate 2.0e-5 \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --eval_strategy "no" \
    --report_to none \
    --output_dir "$SFT_OUT"

ls -la "$SFT_OUT"
test -f "$SFT_OUT/config.json" || { echo "FAIL: SFT did not save a model to $SFT_OUT"; exit 1; }
echo "SFT PASS"

echo ""
echo "=== [2/3] Reward modeling smoke (tiny fixture, 2 steps) ==="
uv run examples/scripts/reward_modeling.py \
    --model_name_or_path "$SFT_OUT" \
    --dataset_name trl-internal-testing/zen \
    --dataset_config standard_preference \
    --learning_rate 1.0e-5 \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --logging_steps 1 \
    --eval_strategy "no" \
    --report_to none \
    --output_dir "$RM_OUT"

ls -la "$RM_OUT"
test -f "$RM_OUT/config.json" || { echo "FAIL: RM did not save a model to $RM_OUT"; exit 1; }
echo "RM PASS"

echo ""
echo "=== [3/3] PPO smoke (tiny fixture, ~2 optimization steps) ==="
uv run examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/zen \
    --dataset_config standard_prompt_only \
    --dataset_test_split test \
    --output_dir "$PPO_OUT" \
    --model_name_or_path "$TINY_MODEL" \
    --sft_model_path "$SFT_OUT" \
    --reward_model_path "$RM_OUT" \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --total_episodes 4 \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 8 \
    --eval_strategy "no" \
    --report_to none
echo "PPO PASS"

echo ""
echo "=== ALL STAGES PASS ==="
echo "Outputs preserved at:"
echo "  SFT: $SFT_OUT"
echo "  RM:  $RM_OUT"
echo "  PPO: $PPO_OUT"
echo "Remove with: rm -rf $SFT_OUT $RM_OUT $PPO_OUT"
