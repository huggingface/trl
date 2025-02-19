export CUDA_VISIBLE_DEVICES=3,4,5,6

accelerate launch --config_file=./mytest.yaml mytest.py 
