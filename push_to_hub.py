import os
# 强制走官方 Hugging Face，禁止镜像
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import argparse
import os


def push_checkpoint_to_hub(
    checkpoint_path: str,
    repo_id: str,
    hf_token: str = None,
    model_type: str = "auto",
    is_private: bool = False,
    commit_message: str = "Upload model checkpoint"
):
    # ===================== 强制登录 =====================
    api = HfApi(token=hf_token)
    print(f"✅ 登录成功，用户：{api.whoami()['name']}")

    # ===================== 【关键修复】强行创建仓库 =====================
    print(f"🔨 正在创建/确认仓库：{repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=is_private,
            exist_ok=True,
            repo_type="model"  # 必须加这个！
        )
        print("✅ 仓库已准备就绪")
    except Exception as e:
        print(f"❌ 仓库创建失败：{e}")
        return

    # ===================== 加载模型（跳过也可以，直接上传文件夹） =====================
    print(f"🔍 加载本地检查点：{checkpoint_path}")

    try:
        config = AutoConfig.from_pretrained(checkpoint_path)
        tokenizer = None
        if os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print("ok")
        model = AutoModel.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"⚠️  模型加载失败，但仍会尝试直接上传文件夹：{e}")

    # ===================== 【最稳】直接上传文件夹 =====================
    print(f"\n🚀 开始上传到：{repo_id}")
    upload_folder(
        repo_id=repo_id,
        folder_path=checkpoint_path,
        token=hf_token,
        commit_message=commit_message,
        repo_type="model",  # 必加！

    )

    print(f"\n🎉 上传成功！https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--repo-id", required=True, type=str)
    parser.add_argument("--hf-token", required=True, type=str)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--model-type", default="auto", choices=["auto", "transformers", "peft"])
    
    args = parser.parse_args()

    push_checkpoint_to_hub(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
        hf_token=args.hf_token,
        model_type=args.model_type,
        is_private=args.private
    )