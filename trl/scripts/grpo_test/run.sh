#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PYTHONPATH:/home/misc/jinpan/trl-jin" # Change it to your own python path
accelerate launch --config_file=trl/scripts/grpo_test/grpo_sgl_test.yaml trl/scripts/grpo_test/grpo_sgl_test.py
