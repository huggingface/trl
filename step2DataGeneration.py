import argparse
import json
from typing import List
import math
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,  # optional: 仅用于“transformers也加载你的模型”
)
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="artemis13fowl/imdb")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--model_name_or_path", type=str, required=True)  # 你的生成模型
    p.add_argument("--output_path", type=str, default="./imdb_pref_tokenized")
    p.add_argument("--max_samples", type=int, default=25000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine", type=str, default="auto", choices=["auto", "vllm", "transformers"])
    return p.parse_args()


def make_5word_prompt(text: str) -> str:
    words = text.strip().split()
    prefix = " ".join(words[:5]) if len(words) >= 5 else " ".join(words)
    return f"Continue this movie review naturally:\n{prefix}"


def batched(iterable, bs):
    for i in range(0, len(iterable), bs):
        yield iterable[i : i + bs]

def log_odds(p:float, eps:float=1e-6) -> float:
    p = min(max(p,eps),1.0-eps)
    return math.log(p/(1.0-p)) 

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))



@torch.no_grad()
def generate_with_transformers(model, tokenizer, prompts, max_new_tokens, temperature, top_p):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    # shape: [B*2, T]
    new_tokens = out[:, prompt_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    resp1, resp2 = [], []
    for i in range(0, len(decoded), 2):
        resp1.append(decoded[i])
        resp2.append(decoded[i + 1] if i + 1 < len(decoded) else "")
    return resp1, resp2


@torch.no_grad()
def sentiment_scores(texts: List[str], tok, model, device: str):
    # 返回正向概率 p_pos
    inputs = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)  # [B,2]
    return probs[:, 1].detach().cpu().tolist()  # positive prob


def main():
    args = parse_args()

    # 1) 数据
    ds = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # text/label
    texts = ds["text"]
    labels = ds["label"]  # 0=neg, 1=pos (IMDB常见定义)
    prompts = [make_5word_prompt(t) for t in texts]

    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = "left"

    use_vllm = args.engine in ["auto", "vllm"]
    llm = None
    gen_model = None

    if use_vllm:
        try:
            llm = LLM(model=args.model_name_or_path)  # trust_remote_code 在这里可省略
        except Exception as e:
            if args.engine == "vllm":
                raise
            print(f"[WARN] vLLM init failed, fallback to transformers. reason: {e}")

    if llm is None:
        gen_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        ).eval()

    all_resp_1, all_resp_2 = [], []
    for batch_prompts in tqdm(list(batched(prompts, args.batch_size)), desc="Generating"):
        if llm is not None:
            sp = SamplingParams(
                n=2,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                seed=args.seed,
            )
            outputs = llm.generate(batch_prompts, sp)
            for out in outputs:
                cands = [x.text for x in out.outputs]
                if len(cands) < 2:
                    cands = (cands + [""])[:2]
                all_resp_1.append(cands[0])
                all_resp_2.append(cands[1])
        else:
            r1, r2 = generate_with_transformers(
                gen_model,
                gen_tokenizer,
                batch_prompts,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            all_resp_1.extend(r1)
            all_resp_2.extend(r2)

    # 4) 用 siebert/sentiment-roberta-large-english 做偏好标注
    reward_name = "siebert/sentiment-roberta-large-english"
    reward_tok = AutoTokenizer.from_pretrained(reward_name)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_model.to(device).eval()

    # 打分时用 prompt+completion 更稳
    pair_1 = [p + "\n" + r for p, r in zip(prompts, all_resp_1)]
    pair_2 = [p + "\n" + r for p, r in zip(prompts, all_resp_2)]

    score1 = []
    score2 = []
    for b1, b2 in tqdm(
        zip(batched(pair_1, args.batch_size), batched(pair_2, args.batch_size)),
        total=(len(pair_1) + args.batch_size - 1) // args.batch_size,
        desc="Scoring",
    ):
        s1 = sentiment_scores(b1, reward_tok, reward_model, device)
        s2 = sentiment_scores(b2, reward_tok, reward_model, device)
        score1.extend(s1)
        score2.extend(s2)

    # label对齐分数：label=1看p_pos, label=0看p_neg=(1-p_pos)
    examples = []
    for i in range(len(prompts)):
        s1 = score1[i]  # s(X, Y1)
        s2 = score2[i]  # s(X, Y2)

        # r*(X,Y)=log(s/(1-s))
        r1 = log_odds(s1)
        r2 = log_odds(s2)

        # P*(Y1 ≻ Y2|X)=sigmoid(r1-r2)
        p_y1_over_y2 = sigmoid(r1 - r2)

        # 硬标签（可直接用于DPO）
        if p_y1_over_y2 >= 0.5:
            chosen_text, rejected_text = all_resp_1[i], all_resp_2[i]
            chosen_reward, rejected_reward = r1, r2
        else:
            chosen_text, rejected_text = all_resp_2[i], all_resp_1[i]
            chosen_reward, rejected_reward = r2, r1

    
        examples.append(
            {
                "prompt": prompts[i],
                "chosen": chosen_text,
                "rejected": rejected_text,
                # 额外保留软标签/奖励，便于分析或后续加权训练
                "bt_prob_y1_over_y2": p_y1_over_y2,
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
            }
        )


    # 保存成HF Dataset（可直接给DataCollatorForPreference）
    out_ds = Dataset.from_list(examples)
    out_ds.save_to_disk(args.output_path)

    # 可选：导出jsonl看样本
    with open(args.output_path + ".jsonl", "w", encoding="utf-8") as f:
        for x in examples[:1000]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Saved tokenized preference dataset to: {args.output_path}")
    print(f"Num examples: {len(out_ds)}")
    pos_chosen = sum(1 for x in examples if x['chosen_reward'] > 0)
    print(f"Chosen 样本中正向影评占比: {pos_chosen/len(examples):.2%}")

if __name__ == "__main__":
    main()