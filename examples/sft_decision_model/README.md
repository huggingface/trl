# Decision model with an SFTTrainer subclass

A decision model, sometimes called a "System One" model, answers typed questions about a state with a probability for every option, from one forward pass and without generating text:

- `noul`: a yes/no question, answered with the probability of `true`.
- `choice`: one option among named options, like routing a ticket to a team.
- `score`: a level on an ordinal scale, like rating urgency.

`transformers serve` answers these questions on its `/v1/systemone` endpoint with any chat model: the prompt describes the question with one label per option (`Yes`/`No`, or `A`, `B`, `C`, ...), and the next-token logits of the labels at the end of the prompt, normalized with a softmax, are the probabilities of the options. This example trains [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) to answer them well, on the exact prompt of the endpoint, so that the trained model is served as is.

Qwen3.5's linear attention layers need the `flash-linear-attention` and `causal-conv1d` kernels, without which they fall back to a reference implementation several times slower:

```sh
pip install flash-linear-attention
pip install causal-conv1d --no-build-isolation  # compiled against the installed torch, needs nvcc
```

## 1. Training

[`1_sft.py`](1_sft.py) is an [`SFTTrainer`] subclass whose `compute_loss` keeps the logits of the labels at the last token of each prompt and computes a cross-entropy with the target probabilities of the options. Three settings make it work:

- `remove_unused_columns=False` and a collator built on `DataCollatorForLanguageModeling` carry the token ids of the labels (`label_ids`) and the targets through to `compute_loss`.
- The collator is padding-free: the prompts of a batch are one sequence, and `logits_to_keep` asks the model for the logits at the last token of each prompt only, instead of the logits of every token over a 248k-token vocabulary.
- The dataset is tokenized and filtered in the script, with `skip_prepare_dataset=True`: preparing it would truncate the prompts at `max_length`, cutting the token the answer is read at.

The data is 40k questions of [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions), from its sources licensed for commercial use: half `choice`, a quarter `noul` and a quarter `score`, with soft targets ([`decisions.py`](decisions.py)). Prompts longer than 512 tokens are left out, which keeps about 30.7k questions. LoRA on all linear layers of the text model, then the adapter is merged:

```sh
python examples/sft_decision_model/1_sft.py
```

On one RTX 4070 Laptop GPU (8 GB): 961 steps of 32 questions, 4.9 s/step, 6 GB peak; one epoch takes 78 minutes. The loss goes from 1.12 to 0.71 and the accuracy on the training questions from 51% to 73%.

The trained model is on the Hub as [rycerzes/Qwen3.5-2B-Decision](https://huggingface.co/rycerzes/Qwen3.5-2B-Decision) (with your own run, use the output directory). Serve it with `transformers serve`, which needs `requests` on top of the `serving` extra:

```sh
pip install "transformers[serving]>=5.20.0" requests
transformers serve
```

```sh
curl http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "rycerzes/Qwen3.5-2B-Decision",
  "state": "Our checkout has been returning 500 errors since this morning and customers cannot pay. We have lost dozens of orders.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team handles this?", "criteria": {"billing": "Charges, invoices and refunds", "engineering": "Bugs and outages", "sales": "New contracts"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "This week", "Today", "Right now"]},
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
  }
}'
```

```json
{
  "team": {"type": "choice", "choice": "engineering", "probabilities": {"billing": 0.106, "engineering": 0.883, "sales": 0.011}, "confidence": 0.825},
  "urgency": {"type": "score", "score": 2.67, "probabilities": {"0": 0.010, "1": 0.032, "2": 0.235, "3": 0.723}, "confidence": 0.671, "legend": {...}},
  "refund": {"type": "noul", "noul": 0.133}
}
```

(the `answers` of the response, rounded)

## 2. Evaluation

[`2_evaluate.py`](2_evaluate.py) sends the requests of two held-out sets to a running `transformers serve`, with all the questions of a state in one request:

- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), test split: 400 states (security incidents, agent traces, invoices, customer service) with 5 questions each, and soft gold probabilities from 3 annotators.
- [JevBench/jevbench](https://huggingface.co/datasets/JevBench/jevbench): 240 states with 8 to 12 questions each, 2,240 questions in all.

None of their 640 states is in the training questions, and none shares a 13-word sequence with them. Accuracy counts the questions whose most probable option is the most probable in the gold, and the Brier score is the squared distance between the probabilities of the options and the gold ones (0 is perfect, 2 the worst).

```sh
transformers serve
python examples/sft_decision_model/2_evaluate.py --model Qwen/Qwen3.5-2B --output base.jsonl
python examples/sft_decision_model/2_evaluate.py --model Qwen3.5-2B-Decision --output trained.jsonl
```

| | typed-decisions accuracy | typed-decisions Brier | JevBench accuracy | JevBench Brier |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B | 45.4% | 0.372 | 73.1% | 0.308 |
| after training | **53.9%** | **0.226** | **77.2%** | **0.239** |

By question type:

| | typed-decisions noul | choice | score | JevBench noul | choice | score |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B | 50.5% | 47.7% | 39.9% | 82.5% | 74.1% | 26.4% |
| after training | 51.2% | 52.5% | 57.1% | 84.6% | 79.3% | 37.3% |

The probabilities improve more than the most probable options: the Brier score drops by 39% on typed-decisions and 22% on JevBench. Ordinal `score` questions gain the most, and remain the weakest type on JevBench.
