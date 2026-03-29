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
from trl import DPOConfig, DPOTrainer, ModelConfig, TrlParser

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """自定义脚本参数"""
    dataset_name: str = field(
        default="openai/summarize_from_feedback", 
        metadata={"help": "数据集名称"}
    )
    cache_dir: str = field(
        default="/root/autodl-tmp/cache", 
        metadata={"help": "数据缓存目录"}
    )
    sanitize_threshold: float = field(
        default=0.0, 
        metadata={"help": "只保留 |score_a - score_b| > threshold 的样本"}
    )

def build_example(ex):
    subreddit = ex["info"]["subreddit"]
    post = ex["info"]["post"]
    title = ex["info"]["title"]
    text1, text2 = ex["summaries"][0]['text'], ex["summaries"][1]['text']
    choice = ex["choice"]

    prompt = f"SUBREDDIT: {subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR: "
    if choice == 0:
        chosen_text, rejected_text = text1, text2
    else:
        chosen_text, rejected_text = text2, text1
    return {
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def get_processed_dataset(script_args, training_args, tokenizer):
    """实现数据持久化缓存逻辑"""
    model_name = training_args.model_name_or_path.split('/')[-1] if hasattr(training_args, 'model_name_or_path') else "model"
    processed_path = os.path.join(script_args.cache_dir, f"dpo_tldr_{model_name}")
    
    # 缓存命中检查
    if os.path.exists(processed_path) and len(os.listdir(processed_path)) > 0:
        logger.info(f" 命中缓存！加载已处理数据集: {processed_path}")
        return load_from_disk(processed_path)
    
    logger.info(" 缓存未命中，开始处理原始数据集...")
    BASE = "https://hf-mirror.com/datasets/openai/summarize_from_feedback/resolve/refs%2Fconvert%2Fparquet"
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{BASE}/comparisons/train/0000.parquet",
            "validation": f"{BASE}/comparisons/validation/0000.parquet",
        },
    )
    
    # 转换格式
    dataset = dataset.map(build_example, num_proc=4)
    
    # 可以在这里加入你的 sanitize_threshold 过滤逻辑
    # if script_args.sanitize_threshold > 0:
    #     dataset = dataset.filter(...)

    # 保存到磁盘
    os.makedirs(script_args.cache_dir, exist_ok=True)
    dataset.save_to_disk(processed_path)
    logger.info(f" 数据集已成功缓存至: {processed_path}")
    
    return dataset

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
    dataset = get_processed_dataset(script_args, training_args, tokenizer)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # 4. 加载模型 (显式开启 Flash Attention 2)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        device_map="auto",
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

if __name__ == "__main__":
    main()