#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="/sgl-workspace/ryang/trl:$PYTHONPATH"
accelerate launch --config_file=trl/scripts/grpo_test/grpo_sgl_test.yaml trl/scripts/grpo_test/grpo_sgl_test.py
