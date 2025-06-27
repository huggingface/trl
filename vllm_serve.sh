python -m trl.scripts.vllm_serve \
  --model /large_storage/goodarzilab/parsaidp/last_kegg_ckpt_fixed/ \
  --use_dna_llm \
  --dna_model_name "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"


ps aux | grep -E "(vllm|python.*vllm_serve)" | grep -v grep