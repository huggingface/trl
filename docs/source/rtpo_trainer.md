# 🚀 Reverse Thinking Policy Optimization (RTPO)

This Work introduces **Reverse Thinking Policy Optimization (RTPO)** — a new RL training method for LLMs built on top of `GRPOTrainer`.
---

## 🔍 Motivation

Current GRPO-based RL methods require the model to autonomously generate a full chain-of-thought before producing the final answer.
However, many training datasets already contain **complete, high-quality reasoning traces** that the model could benefit from.

RTPO is designed to:

* Utilize existing reasoning traces as **auxiliary CoT** to support early-stage rollouts.
* Force the model to gradually reconstruct its own reasoning by **shortening the auxiliary CoT** step by step.
* Enable a *reverse learning schedule*: the model first learns to output correct answers, then progressively learns how to reason.

---

## 🧠 Method Overview

RTPO modifies the standard GRPO rollout process:

### **Full Auxiliary CoT Injection**

At rollout step 0, the full reasoning chain from the dataset is concatenated into the input prompt.

Model behavior:

* Only needs to generate the *final answer*.
* Benefits from a high-quality reasoning scaffold.

### **Reverse Annealing of Auxiliary CoT**

As training steps increase, RTPO gradually **removes tokens from the end** of the auxiliary CoT based on a configurable schedule:

```
full_reasoning → partial_reasoning → short_hint → empty
```

Expected Model behavior:

* "Fill in" the removed reasoning process.
* Learns to produce longer reasoning as annealing progresses.

### **Interesting Finding: Emergent Shorter Reasoning**

Unexpectedly, RTPO also teaches the model to **shorten its reasoning**:

* When the model does not regenerate the removed tokens, and instead directly outputs the correct final answer,
* Over training, the model consistently generates **shorter, more efficient reasoning chains**.

---

More experiments are ongoing and will be included later.


---

## 🧪 Example Usage
install developing version trl from https://github.com/huggingface/trl
Check `train_rtpo.py`

