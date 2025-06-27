# DNA-Enhanced vLLM Server Testing Guide

This guide will help you test the modified vLLM server with DNA functionality using both simple and KEGG dataset examples.

## Prerequisites

1. **Install required dependencies:**
   ```bash
   pip install trl vllm fastapi uvicorn pydantic requests
   # For DNA processing (if available):
   pip install bioreason datasets
   ```

2. **Ensure you have:**
   - A trained LLM model (e.g., Qwen, Llama)
   - Access to a DNA model (e.g., nucleotide-transformer)
   - GPU with sufficient memory

## Quick Test Setup

### Step 1: Start the DNA-Enhanced vLLM Server

```bash
# Basic text-only server (no DNA)
python -m trl.scripts.vllm_serve \
  --model /path/to/your/llm/model \
  --host localhost \
  --port 8000 \
  --gpu_memory_utilization 0.8

# DNA-enhanced server
python -m trl.scripts.vllm_serve \
  --model /path/to/your/llm/model \
  --host localhost \
  --port 8000 \
  --gpu_memory_utilization 0.8 \
  --use_dna_llm \
  --dna_model_name "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" \
  --max_length_dna 2048
```

### Step 2: Run Simple Tests

```bash
# Test the server with simple examples
python simple_test_dna_vllm.py --host localhost --port 8000
```

## Advanced Testing with KEGG Dataset

### Example Usage

If you have the bioreason components and KEGG dataset access:

```bash
# Advanced testing with real KEGG data
python test_dna_vllm_serve.py \
  --llm_dir /path/to/your/llm/model \
  --dna_model_name "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" \
  --enable_dna \
  --num_examples 5 \
  --kegg_data_dir_huggingface "wanglab/kegg"
```

## Manual API Testing

You can also test the API directly using curl or Python requests:

### Text-Only Generation

```bash
curl -X POST "http://localhost:8000/generate/" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["What is DNA?"],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### DNA+Text Generation

```bash
curl -X POST "http://localhost:8000/generate/" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Analyze this DNA: <|dna_start|><|dna_pad|><|dna_end|> What does this sequence do?"],
    "dna_sequences": [["ATGAAGGCCCCCCTGCTG"]],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Server Configuration Options

### Basic Parameters
- `--model`: Path to your LLM model
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 8000)
- `--gpu_memory_utilization`: GPU memory usage (default: 0.9)
- `--max_model_len`: Maximum model context length

### DNA-Specific Parameters
- `--use_dna_llm`: Enable DNA functionality
- `--dna_model_name`: DNA model to use
- `--max_length_dna`: Maximum DNA sequence length
- `--dna_is_evo2`: Set if using Evo2 model
- `--dna_embedding_layer`: Specific layer for Evo2

### Parallelization
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism
- `--data_parallel_size`: Number of parallel workers

## What Gets Loaded

### DNA Projection Components
The server will automatically load your trained DNA projection weights:

```
🧬 Initializing DNAEmbeddingProcessor...
  DNA model: InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
  Text model: /path/to/your/model
  Evo2: False
🔧 Loading trained DNA projection weights from /path/to/your/model/dna_projection.pt
✅ Trained DNA projection weights loaded successfully
✅ DNAEmbeddingProcessor initialized successfully
```

### Components Replicated from DNALLMModel:
1. **DNA Projection Layer**: `nn.Linear(dna_hidden_size, text_hidden_size)`
2. **Chat Template**: Uses the same CHAT_TEMPLATE as your DNALLMModel
3. **DNA Tokens**: `<|dna_start|>`, `<|dna_pad|>`, `<|dna_end|>` 
4. **DNA Processing**: Exact same `process_dna_embeddings()` logic
5. **DNA Integration**: Same embedding replacement logic as `DNALLMModel.generate()`

### What Happens During Generation:
```
🧬 Processing DNA embeddings for 2 sequences
🧬 Using standard HuggingFace DNA model
🧬 Raw DNA hidden states shape: torch.Size([2, 512, 512])
🧬 Projected DNA states shape: torch.Size([2, 512, 4096])
🧬 Found 4 DNA tokens in text
🧬 Generated 4 DNA features
🧬 Before DNA replacement - text embeds mean: 0.0234
🧬 After DNA replacement - text embeds mean: 0.0891
🧬 DNA successfully integrated into text embeddings!
```

## Expected Outputs

### Successful Text Generation
```json
{
  "completion_ids": [[101, 102, 103, ...]]
}
```

### DNA Generation Success Indicators
- Server starts without errors
- DNA projection weights loaded from `dna_projection.pt`
- Health check returns 200
- Text-only generation works
- DNA+text generation processes DNA sequences
- DNA embeddings properly integrated into text
- Returns completion token IDs

### Common Issues & Solutions

1. **Server won't start:**
   - Check model path exists
   - Verify GPU memory availability
   - Ensure all dependencies installed

2. **DNA functionality disabled:**
   - Verify `--use_dna_llm` flag
   - Check DNA model name is correct
   - Ensure bioreason components installed

3. **Out of memory errors:**
   - Reduce `--gpu_memory_utilization`
   - Decrease `--max_model_len`
   - Use smaller DNA sequences

## Performance Benchmarking

The test scripts will measure:
- **Generation time** per example
- **Tokens per second** throughput  
- **Success rate** for DNA processing
- **Memory usage** patterns

### Example Results
```
🏁 TEST RESULTS SUMMARY
====================================
📊 Text-Only Generation:
  ✅ Status: SUCCESS
  ⏱ Time: 1.23s
  📊 Completions: 2

🧬 DNA+Text Generation:
  📈 Total Examples: 2
  ✅ Successful: 2
  ❌ Failed: 0
  ⏱ Total Time: 4.56s
  📊 Avg Time/Example: 2.28s
  📈 Success Rate: 100.0%
```

## API Endpoints

### Available Endpoints
- `GET /health/` - Health check
- `POST /generate/` - Text/DNA generation
- `POST /init_communicator/` - Initialize weight sync
- `POST /update_named_param/` - Update model weights
- `POST /reset_prefix_cache/` - Reset prefix cache

### Generate Endpoint Parameters
- `prompts`: List of text prompts
- `dna_sequences`: List of DNA sequence lists per prompt (optional)
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter

## Troubleshooting

### Debug Mode
Add these flags for debugging:
```bash
--log_level debug
```

### Check Server Logs
Monitor server output for:
- DNA model loading status
- Memory usage warnings
- Generation errors

### Validate DNA Processing
- Confirm DNA sequences are properly tokenized
- Verify embedding dimensions match
- Check DNA token placement in prompts

## Next Steps

1. **Start with simple test** to verify basic functionality
2. **Try real KEGG data** if available
3. **Benchmark performance** vs standard models
4. **Scale up** with more examples and longer sequences

The DNA-enhanced vLLM server should provide efficient multimodal DNA+text generation while maintaining vLLM's performance benefits! 