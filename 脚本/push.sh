
set -e
python push_to_hub.py \
  --checkpoint /root/autodl-tmp/Qwen2-1.5B-SFT-IF/checkpoint-1392\
  --repo-id chenyongxi/Qwen2-1.5B-SFT-IF \
  --hf-token 


