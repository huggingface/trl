import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed
)
from trl import DPOConfig, DPOTrainer, ModelConfig, TrlParser,ScriptArguments
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. 使用 TrlParser 解析三个 dataclass
    # ModelConfig: 包含 model_name_or_path, trust_remote_code 等
    # DPOConfig: 继承自 TrainingArguments，包含 output_dir, bf16, lr 等
    # ScriptArguments: 你的自定义业务参数
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # 设置随机种子
    set_seed(training_args.seed)

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载并缓存数据集
    dataset = load_dataset(script_args.dataset_name).select_columns(["chosen","rejected"])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 4. 加载模型 (显式开启 Flash Attention 2)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # 强制开启以追求 48h 目标
        trust_remote_code=model_args.trust_remote_code,
        local_files_only=True, # 确保使用本地缓存
    )

    # 5. 初始化 DPOTrainer
    # 注意：TRL 的 DPOTrainer 会自动处理 ref_model，如果未传则会自动克隆一份
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 6. 开始训练
    logger.info("*** 开始 DPO 训练 ***")
    trainer.train()
    
    # 7. 保存最终产物
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f" 训练完成。模型已保存至 {training_args.output_dir}")
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        logger.info("推送至hub")

if __name__ == "__main__":
    main()