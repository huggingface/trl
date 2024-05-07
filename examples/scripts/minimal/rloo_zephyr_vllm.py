import multiprocessing
import shutil

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig
from trl.trainer.rloo_trainer_vllm import RLOOConfig, RLOOTrainer


"""
python -i examples/scripts/minimal/rloo_zephyr_vllm.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/rloo_zephyr_vllm \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 8 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped  \
    --reward_model_path EleutherAI/pythia-1b-deduped  \
    --non_eos_penalty \
    --truncate_token eos \
    --response_length 512 \

# run REINFORCE w/ RLOO; `--epochs 1 --num_mini_batches 1` in PPO is equivalent to REINFORCE
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.7.yaml \
    examples/scripts/minimal/rloo_zephyr_vllm.py \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --rloo_k 2 \
    --learning_rate 3e-6 \
    --output_dir models/minimal/rloo_zephyr_vllm_k2_seed1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 8 \
    --total_episodes 200000 \
    --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta \
    --sft_model_path HuggingFaceH4/mistral-7b-sft-beta \
    --reward_model_path weqweasdas/RM-Mistral-7B \
    --deepspeed3 \
    --kl_coef 0.15 \
    --non_eos_penalty \
    --truncate_token eos \
    --response_length 1024 \
"""


if __name__ == "__main__":
    parser = HfArgumentParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_config_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        # a default chat template to simply concatenate the messages
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        attn_implementation="flash_attention_2",
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        attn_implementation="flash_attention_2",
    )
    ################
    # Dataset
    ################
    raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1000))
    train_dataset = raw_datasets["train_sft"]
    eval_dataset = raw_datasets["test_sft"]

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=1 if config.sanity_check else multiprocessing.cpu_count(),
            load_from_cache_file=not config.sanity_check,
        )

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    # filtering
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 1024)
    eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 1024)
    ################
    # Training
    ################
    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    trainer.push_to_hub()
    trainer.generate_completions(True)
