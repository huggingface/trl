# Fuse three RL experts into one model with MOPD

Say you trained one RL expert per domain, for math, code and instruction following, all from the same SFT checkpoint, and you want one model that keeps what each of them learned. Multi-teacher on-policy distillation (MOPD) does that. The student answers prompts from every domain, and the expert of each prompt's domain scores the answer token by token. This example runs it with [`AsyncDistillationTrainer`](https://huggingface.co/docs/trl/async_distillation_trainer) and the experts released with [Open-MOPD](https://huggingface.co/papers/2608.19098).

## 1. Where the student starts

Open-MOPD released its whole recipe on SmolLM3-3B: an SFT checkpoint on mixed data, [`Open-MOPD-SmolLM3-3B-MixSFT`](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT), and three experts trained from it with GRPO, for [math](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math), [code](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Code) and [instruction following](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-IF). The SFT checkpoint is the student, so this is the fusion stage of their recipe. With the Open-MOPD evaluation sets and scorers, it starts at 17.5 on AIME24, 19.2 on AIME25 and 67.5 on IFEval.

All four models share one tokenizer, which the trainer needs, since it compares the student and the teacher token by token.

## 2. Send each prompt to its teacher

Each teacher is a vLLM server, and the trainer gets one URL per teacher. A `teacher_id` column says which one scores each prompt. The [Open-MOPD prompt mix](https://huggingface.co/datasets/BytedTsinghua-SIA/Open-MOPD-Data) already labels every prompt with its domain, so the column is a rename. About half the mix is instruction following, so [`async_distillation_mopd.py`](async_distillation_mopd.py) samples math, code and instruction following 40/40/20:

```python
dataset = load_dataset("BytedTsinghua-SIA/Open-MOPD-Data", data_files="rl_prompt_mix/train.parquet", split="train")
dataset = dataset.select_columns(["prompt", "domain"]).rename_column("domain", "teacher_id").shuffle(seed=42)
dataset = interleave_datasets(
    [dataset.filter(lambda row, d=domain: row["teacher_id"] == d) for domain in ("math", "code", "if")],
    probabilities=[0.4, 0.4, 0.2],
    seed=42,
)

config = AsyncDistillationConfig(
    teacher_server_urls={"math": "http://localhost:8001", "code": "http://localhost:8002", "if": "http://localhost:8003"},
    beta=1.0,  # reverse KL against the routed teacher
    teacher_top_k=1,  # with beta=1.0 only the sampled token's teacher logprob is used
    ...
)
```

## 3. Run it

MOPD needs five processes at once: the student's vLLM server, which generates and receives the new weights after every step, one vLLM server per teacher, which only score, and the training process. On 4 H200s:

| GPU | What runs on it |
| --- | --- |
| 0 | The student's vLLM server |
| 1 | The math and instruction-following teachers |
| 2 | The code teacher, alone because code prompts are the longest |
| 3 | Training |

Every teacher gets a larger `--max-model-len` than the student, because scoring sends the student's whole answer plus one token, and the two teachers on GPU 1 start one after the other, because vLLM sizes its memory from what is free when it starts. [`launcher.py`](launcher.py) does all of it on Hugging Face Jobs, with one command:

```sh
hf jobs uv run --flavor h200x4 --image huggingface/trl --secrets HF_TOKEN --timeout 6h \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_distillation_mopd/launcher.py
```

The `vllm serve` commands to run it on your own GPUs are in the script's docstring.

## 4. Watch the token imbalance

The trainer logs two metrics per teacher: `teacher_token_frac/<id>`, the share of each step's tokens that went to it, and `teacher_jsd/<id>`, how far the student is from it. Keep an eye on `teacher_token_frac`. The loss is averaged over tokens, so each domain counts in proportion to the tokens it produces. Math and code answers run to thousands of tokens, instruction-following ones to a few hundred, so with 20% of the prompts instruction following got 1 to 3% of the tokens, and the student barely moved toward that teacher:

![Per-teacher token share and divergence](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trl/mopd-per-teacher-metrics.png)

The Open-MOPD paper explains this imbalance, together with a fix.

## 5. Measure again

We stopped the run at step 60 (the script runs 90), about 1% of the paper's training, and evaluated it next to the starting point and the paper's final model:

| Model | AIME24 | AIME25 | IFEval |
| --- | --- | --- | --- |
| MixSFT (starting point) | 17.5 | 19.2 | 67.5 |
| + MOPD with TRL, 60 steps ([model](https://huggingface.co/sergiopaniego/open-mopd-smollm3-trl-naive)) | 24.2 | 22.5 | 73.0 |
| Open-MOPD final model (paper) | 19.2 | 25.0 | 75.8 |

AIME with 4 samples per problem, so a few points either way is noise, and IFEval strict prompt-level accuracy. Code was not evaluated. On math the student is close to the paper's final model, and on IFEval it stays 2.8 points below, in the domain that got the fewest tokens. The metrics of the run are on [Trackio](https://huggingface.co/spaces/sergiopaniego/open-mopd-trl).

## 6. Use your own experts

The example uses math, code and instruction following because those are the experts Open-MOPD released. With your own RL experts, trained from the same base model so they share a tokenizer, serve each one with vLLM, add its URL to `teacher_server_urls` and label each prompt with its domain. To close the instruction-following gap, train longer or give those prompts more weight, and the per-teacher metrics will tell you if it helps.

The [MOPD blog post](https://huggingface.co/blog/sergiopaniego/mopd-trl) covers which labs use MOPD and how, and the checkpoint, the dashboard and the Open-MOPD models and data are in [this collection](https://huggingface.co/collections/sergiopaniego/mopd-with-trl-experiment).
