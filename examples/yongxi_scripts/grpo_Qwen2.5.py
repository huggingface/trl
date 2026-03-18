# GroupRelativePolicyOptimization with LoRA/QLoRA using TRL

# LoRA + Quant + Liger + Paged_adamw_8bit + sdpa = 7B模型的训练变得可行

# 1、安装依赖
!pip install -Uq "trl[peft]" bitsandbytes trackio math_verify liger_kernel

from huggingface_hub import notebook_login

notebook_login()

# 2、载入数据集

from datasets import load_dataset

# 制定训练集
dataset_name = 'AI-MO/NuminaMath-TIR'
train_dataset = load_dataset(dataset_name, split = 'train[:5%]')

print(train_dataset[0])

# 系统提示词
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant  "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process is enclosed strictly within <think> and </think> tags. "
    "After closing </think>, the assistant MUST provide the final answer in plain text."
)

# 将每一个实例的格式转换成对话格式,返回的键值汇入原字典
def make_conversation(example):
    return {
        "prompt":[
            {"role": "system","content": SYSTEM_PROMPT},
            {"role": "user", "content":example['problem']}
        ],
    }
train_dataset = train_dataset.map(make_conversation)
# 移除掉不用的列：原对话信息和问题内容，保留solution
train_dataset = train_dataset.remove_columns(['messages','problem'])


# 3、载入模型，配置LoRA/QLoRA
model_id, output_dir = "Chenyongxi/Qwen2-0.5B/SFT", ""



