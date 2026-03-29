
set -e
python push_to_hub.py \
  --checkpoint /root/autodl-tmp/TLDR/checkpoint-500\
  --repo-id chenyongxi/DPO_TLDR_1B_checkpoint_500 \
  --hf-token 