from datasets import load_dataset
from trl import RewardTrainer,RewardConfig
import argparse

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
    p.add_argument("--output_path", type=str, default="./HH-SFT")
    p.add_argument("--exp_name", type=str, default="SFT")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()



def main():
    args = parse_args()

    train_dataset = load_dataset(args.dataset_name,split = "train")
    test_dataset = load_dataset(args.dataset_name, split = "test")

    config = RewardConfig(
        output_dir = args.output_path,
        num_train_epochs = args.epochs,
        logging_steps=10,
        report_to = "wandb",
        run_name = args.exp_name,
        save_strategy = "epoch",
        save_only_model = True,
        bf16 = True,


    )

    trainer = RewardTrainer(
        model=args.model_name_or_path,
        train_dataset= train_dataset,
        args = config,
        eval_dataset = test_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()