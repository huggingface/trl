# Constrained Generative Policy Optimization Trainer

## Overview

Constrained Generative Policy Optimization (CGPO) was proposed in [The Perfect Blend: Redefining RLHF with Mixture of Judges](https://huggingface.co/papers/2306.13649) by Tengyu Xu, Eryk Helenowski, Karthik Abinav Sankararaman, Di Jin, Kaiyan Peng, Eric Han, Shaoliang Nie, Chen Zhu, Hejia Zhang, Wenxuan Zhou, Zhouhao Zeng, Yun He,Karishma Mandyam, Arya Talabzadeh, Madian Khabsa, Gabriel Cohen, Yuandong Tian, Hao Ma, Sinong Wang and Han Fang. 

The abstract from the paper is the following:

> Reinforcement learning from human feedback (RLHF) has become the leading approach for fine-tuning large language models (LLM). However, RLHF has limitations in multi-task learning (MTL) due to challenges of reward hacking and extreme multi-objective optimization (i.e., trade-off of multiple and/or sometimes conflicting objectives). Applying RLHF for MTL currently requires careful tuning of the weights for reward model and data combinations. This is often done via human intuition and does not generalize. In this work, we introduce a novel post-training paradigm which we called Constrained Generative Policy Optimization (CGPO). The core of CGPO is Mixture of Judges (MoJ) with cost-efficient constrained policy optimization with stratification, which can identify the perfect blend in RLHF in a principled manner. It shows strong empirical results with theoretical guarantees, does not require extensive hyper-parameter tuning, and is plug-and-play in common post-training pipelines. Together, this can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives.
Our results show that CGPO consistently outperforms other commonly used SoTA RLHF algorithms (such as PPO and DPO) on a wide range of tasks – general chat, STEM questions, instruction following, math, coding and knowledge. In particular, CGPO improves over PPO by 7.4% in AlpacaEval-2 (general chat), 12.5% in Arena-Hard (STEM & reasoning), 2% in IFEval (Instrcution Following), 2% in both MATH and GSM8K (Math & reasoning), 5% in HumanEval (Coding), and 2% in the ARC challenge (Knowledge). We also observe that PPO is susceptible to severe reward hacking behaviors (it exhibits severe regression in popular coding benchmarks) which can be addressed by CGPO. CGPO represents a breakthrough in RLHF, simultaneously addressing reward-hacking and extreme multi-objective optimization, and thereby advancing the state-of-the-art in aligning general-purpose LLMs.


CGPO is designed to address the challenges of reward hacking and the complexities of multi-task learning in RLHF. It introduces three key innovations:
1. A 'Mixture of Judges' (MoJs) combining rule-based and LLM-based judges to collaboratively detect reward hacking and ensure adherence to task-specific constraints.
2. Task-specific optimization strategies (independent MoJs, optimizers and reward models).
3. Three new constrained RLHF optimizers: Calibrated-Regularized Policy Gradient (CRPG), Constrained Online Direct Preference Optimization (CODPO), and Calibrated-Regularized Reward Ranking Finetuning (CRRAFT)

This post-training method was contributed by [Gaetan Lopez](https://github.com/gaetanlop) + Add the names of the future PR reviewers (kashif, lewton, qgallouedec?)

> [!WARNING]
> The `CGPOTrainer` currently only supports the single task with single objective setting. CGPO in multi-tasks with multi-objectives will be added in a future release.

## Usage tips

The `CGPOTrainer` is a wrapper around the transformers [`Trainer`] class that takes in a reward model and a mixture of judges. It mostly requires three parameters to be set via the [`CGPOConfig`] namely:
* `rlhf_optimizer`: specifies the optimizer to use for policy optimization, with three possible options: `crpg`, `codpo` and `crraft`.
* `k`: defines the number of generations per prompt.
* `kl_threshold`: sets the maximum allowable KL divergence between the model and the reference model for each generated completion.

Based on the paper findings: For tasks requiring precise judges and extensive exploration, such as instruction following, math, and coding, use higher values for `k` and a more lenient KL threshold. Conversely, for tasks with less precise judges and where exploration is less critical, such as "general chat",  use lower values of `k` and a stricter KL threshold.

The basic API is as follows:

```python
from datasets import Dataset
from trl import CGPOConfig, CGPOTrainer, MixtureOfConstraintJudges
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

NUM_DUMMY_SAMPLES = 100

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
reward_model = AutoModelForSequenceClassification.from_pretrained("trl-lib/Qwen2-0.5B-Reward", num_labels=1)
mixture_of_judges = MixtureOfConstraintJudges([CustomJudge1, CustomJudge2])

train_dataset = Dataset.from_dict(
    {
        "messages": [
            [
                {"role": "user", "content": "Hi, how are you?"},
                {"role": "assistant", "content": "I'm great thanks"},
            ]
        ]
        * NUM_DUMMY_SAMPLES
    }
)
eval_dataset = Dataset.from_dict(
    {
        "messages": [
            [
                {"role": "user", "content": "What colour is the sky?"},
                {"role": "assistant", "content": "The sky is blue"},
            ]
        ]
        * NUM_DUMMY_SAMPLES
    }
)

training_args = CGPOConfig(
    output_dir="cgpo-model", 
    per_device_train_batch_size=2,
    k=4,
    rlhf_optimizer="crpg",
    kl_threshold=5.,
    )
trainer = CGPOTrainer(
    model=model,
    reward_model=teacher_model,
    mixture_of_judges=mixture_of_judges,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

### ⚠️ Use the same chat template

Make sure that the SFT model and reward model use the _same_ chat template. Otherwise, you may find the model completions are scored incorrectly during training.

### Expected dataset format

The dataset should be formatted as a list of "messages" where each message is a list of dictionaries with the following keys:
* `role`: either `system`, `assistant` or `user`
* `content`: the message content


## CGPOTrainer

[[autodoc]] CGPOTrainer

## CGPOConfig

[[autodoc]] CGPOConfig
