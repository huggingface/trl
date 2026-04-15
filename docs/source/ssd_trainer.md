# SSD

Simple Self-Distillation (SSD) is described in [Embarrassingly Simple Self-Distillation Improves Code Generation](https://huggingface.co/papers/2604.01193).

SSD samples completions from the model at a training-time temperature and truncation configuration, then fine-tunes on those raw, unverified samples with standard cross-entropy loss. It requires no reward model, verifier, teacher model, or reinforcement learning — only a set of problem prompts and the model itself.

In the current TRL implementation:

- the model generates completions at a specified training-time temperature (`temperature`) and truncation (`top_k`, `top_p`)
- the dataset only requires a `prompt` column
- training uses standard cross-entropy loss on the generated completions
- empty or single-line stub completions are filtered by default (`filter_empty=True`)
- the evaluation-time temperature and truncation are set independently at inference time
- vLLM can be used for faster generation via `use_vllm=True` (see [vLLM integration](vllm_integration))

## Usage

```python
from datasets import Dataset

from trl.experimental.ssd import SSDConfig, SSDTrainer

dataset = Dataset.from_dict(
    {
        "prompt": [
            [{"role": "user", "content": "Write a function to add two numbers."}],
            [{"role": "user", "content": "Write a function to check if a number is prime."}],
        ],
    }
)

training_args = SSDConfig(
    output_dir="ssd-model",
    temperature=0.6,           # T_train from the paper
    top_k=20,                  # training-time top-k truncation
    top_p=0.95,                # training-time top-p truncation
    max_completion_length=65536,
    learning_rate=5e-6,
)

trainer = SSDTrainer(
    model="Qwen/Qwen3-4B-Instruct",
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

## Expected dataset columns

Each example must provide:

- `prompt`: the problem prompt (string or conversational format)

No `privileged_context`, reward functions, or teacher model are needed.

## Key hyperparameters

The paper identifies the following key hyperparameters:

- **`temperature`**: training-time sampling temperature (T_train). Higher values create more diverse samples but may include more noise. The paper uses T_train=0.6 with truncation.
- **`top_k`** and **`top_p`**: training-time truncation parameters (rho_train). These suppress low-probability distractor tails during data synthesis.
- **T_eval**: the evaluation-time decoding temperature is set independently at inference time. The paper shows that T_train and T_eval compose through an effective temperature T_eff = T_train * T_eval, with a broad optimal band.

## Example script                                                                                                            
 
Use [`trl/experimental/ssd/ssd.py`](https://github.com/huggingface/trl/blob/main/trl/experimental/ssd/ssd.py) to launch SSD training from the command line. The script supports any causal LM from the Hub, custom local datasets via `--dataset_path`, and PEFT/LoRA via the standard `ModelConfig` flags.

```bash
python trl/experimental/ssd/ssd.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_name microsoft/rStar-Coder \
    --dataset_config seed_sft \
    --prompt_column question \
    --output_dir outputs/ssd-qwen3-4b \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --max_prompt_length 1024 \
    --max_completion_length 65536 \
    --temperature 1.6 \
    --top_k 20 \
    --top_p 0.8 \
    --num_train_epochs 1 \
    --bf16 \
    --report_to trackio
```

## Evaluation on LiveCodeBench

Use [`trl/experimental/ssd/ssd_eval.py`](https://github.com/huggingface/trl/blob/main/trl/experimental/ssd/ssd_eval.py) to evaluate a base model or an SSD-trained checkpoint on LiveCodeBench v6. The script uses vLLM for generation and LiveCodeBench's official `codegen_metrics` for sandboxed `pass@k` scoring; default decoding parameters match Table 3 of the paper.

```bash
python trl/experimental/ssd/ssd_eval.py \
    --model_name_or_path <path-or-repo> \
    --temperature 1.1 --top_k 20 --top_p 0.8 \
    --n 1 \
    --output_file outputs/lcb_v6.json
```

## SSDConfig

[[autodoc]] experimental.ssd.SSDConfig

## SSDTrainer

[[autodoc]] experimental.ssd.SSDTrainer
    - train
    - save_model
    - push_to_hub
