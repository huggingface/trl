from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,pipeline
import os
import numpy as np


os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"

# ======================
# 1. 原始数据预处理（只提取 prompt + completion）
# ======================


# ======================
# 2. 使用 chat_template 拼接并 tokenize
# ======================
def tokenize_and_calculate(example, tokenizer, max_length=8192):
    # 构造标准对话格式（Qwen2 官方格式）
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]}
    ]

    # 拼接 + tokenize（整句输入模型）
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        max_length=max_length,
        truncation=True
    )

    seq_len = len(tokenized["input_ids"])
    return {"length": seq_len}

# ======================
# 3. 统计长度分布
# ======================
def print_length_stats(lengths, split_name="train"):
    lengths = np.array(lengths)
    print(f"\n==================== 【{split_name} 集 Token 长度统计】 ====================")
    print(f"总样本数        : {len(lengths)}")
    print(f"平均 token 数    : {lengths.mean():.2f}")
    print(f"最大 token 数    : {lengths.max()}")
    print(f"最小 token 数    : {lengths.min()}")
    print(f"中位数 token 数  : {np.median(lengths)}")
    print(f"95% 分位长度     : {np.percentile(lengths, 95):.0f}")
    print(f"99% 分位长度     : {np.percentile(lengths, 99):.0f}")
    print(f"建议 max_seq_len : {int(np.percentile(lengths, 99))}")
    print("=" * 80)

# ======================
# 主函数
# ======================
def main():

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    pipe = pipeline("text-generation", model= model_id,max_new_tokens=1024)
    def preprocess(dataset):
        def process(ex):
            prompt = ex["prompt"]
            completion = ex["chosen"][-1]["content"]
            prompt = tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
            return {
                "prompt": prompt,
                "completion": completion
            }

        def process_split(split):
            origin = dataset[split]
            origin = origin.map(process, remove_columns=origin.column_names)
            return origin

        return DatasetDict({
            split: process_split(split) for split in dataset.keys()
        })
    # 1. 加载数据
    ds = load_dataset("BAAI/Infinity-Preference")
    example = ds["train"][0]["prompt"]
    print(example)
    response = pipe(example,skip_special_tokens=False)
    print(response[0]["generated_text"])
    
    

    # # 3. 打印分词器基础信息
    # print("=== tokenizer ===")
    # print("chat_template exists:", bool(getattr(tok, "chat_template", None)))
    # print("eos_token:", tok.eos_token, "eos_token_id:", tok.eos_token_id)
    # print("pad_token:", tok.pad_token, "pad_token_id:", tok.pad_token_id)

    # print("\n=== config ===")
    # print("config.eos_token_id:", getattr(cfg, "eos_token_id", None))
    # print("config.pad_token_id:", getattr(cfg, "pad_token_id", None))

    # print("\n=== special tokens map ===")
    # print(tok.special_tokens_map)

    # ======================
    # 🔥 核心：对每条数据拼接 + tokenize + 统计长度
    # ======================
    #print("\n\n开始 tokenize 并统计长度...")

    # 对 train 集处理
    #tokenized_train = ds["train"].map(
    #     lambda x: tokenize_and_calculate(x, tok),
    #     num_proc=4  # 多线程加速
    # )
    #train_lengths = [d["length"] for d in tokenized_train]

    # 对 test 集处理
    # #tokenized_test = ds["test"].map(
    #     lambda x: tokenize_and_calculate(x, tok),
    #     num_proc=4
    # )
    #test_lengths = [d["length"] for d in tokenized_test]

    # ======================
    # 输出统计结果
    # ======================
    # print_length_stats(train_lengths, "train")
    # print_length_stats(test_lengths, "test")
    


if __name__ == "__main__":
    main()
