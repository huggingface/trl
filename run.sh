export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --config_file=./mytest.yaml mytest.py 
