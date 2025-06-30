python -m trl.scripts.vllm_serve \
  --model /large_storage/goodarzilab/parsaidp/last_kegg_ckpt_fixed/ \
  --use_dna_llm \
  --dna_model_name "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"


ps aux | grep -E "(vllm|python.*vllm_serve)" | grep -v grep

python trl/trl/scripts/vllm_serve.py \
    --model /large_storage/goodarzilab/parsaidp/last_cafa_1.7B_ESM3 \
    --use_protein_llm \
    --protein_model_name esm3_sm_open_v1 \
    --max_length_protein 2048 \
    --gpu_memory_utilization 0.8 \
    --port 8000