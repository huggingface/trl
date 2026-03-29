from trl import SFTTrainer,SFTConfig
from datasets import load_dataset
import os
import argparse
# 1. 设置 Hugging Face 镜像源（加速下载）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 设置 Hugging Face 各类缓存目录到 /root/autodl-tmp
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface/models"
os.environ["DATASETS_CACHE"] = "/root/autodl-tmp/huggingface/datasets"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/huggingface/datasets"



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="artemis13fowl/imdb")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--model_name_or_path", type=str, required=True)  # 你的生成模型
    p.add_argument("--output_path", type=str, default="./HH-SFT-0.5B-Qwen")
    p.add_argument("--exp_name", type=str, default="SFT0325")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--attn_implementation", type = str, default = "flash_attention_2")
    p.add_argument("--save_only_model",type =bool, default=True )
    return p.parse_args()


def merge_dataset(dataset,args):
    chosen_dataset = dataset.map(lambda x:{"text":x['chosen']}, remove_columns= dataset.column_names)
    reject_dataset = dataset.map(lambda x:{"text":x['rejected']},remove_columns= dataset.column_names)
    final_dataset = concatenate_datasets([chosen_dataset,reject_dataset])
    final_dataset = final_dataset.shuffle(seed=args.seed)
    print(final_dataset[len(dataset)+1])
    return final_dataset

from datasets import concatenate_datasets
def main():
    args = parse_args()
    train_dataset = load_dataset(args.dataset_name,split = "train")
    test_dataset = load_dataset(args.dataset_name, split = "test")

    train_dataset = merge_dataset(train_dataset,args)
    test_dataset = merge_dataset(test_dataset,args)
    

    



    training_args = SFTConfig(
        output_dir= args.output_path,
        num_train_epochs=args.epochs,          # 训练轮数
        logging_steps=50,
        save_only_model = True,
        save_steps = 1000,
         # wandb
        report_to="wandb",           # 或 ["wandb"]
        run_name=args.exp_name,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=3e-6,
        lr_scheduler_type="cosine",
     )


    trainer = SFTTrainer(
        model = args.model_name_or_path,
        train_dataset = train_dataset,
        args = training_args,
        eval_dataset = test_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()