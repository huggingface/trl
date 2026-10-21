# Decision model with an SFTTrainer subclass

A decision model, sometimes called a "System One" model, answers typed questions about a state with a probability for every option, from one forward pass and without generating text:

- `noul`: a yes/no question, answered with the probability of `true`.
- `choice`: one option among named options, like routing a ticket to a team.
- `score`: a level on an ordinal scale, like rating urgency.

`transformers serve` answers these questions on its `/v1/systemone` endpoint with any chat model: the prompt describes the question with one label per option (`Yes`/`No`, or `A`, `B`, `C`, ...), and the next-token logits of the labels at the end of the prompt, normalized with a softmax, are the probabilities of the options. This example trains [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) to answer them well, on the exact prompt of the endpoint, so that the trained model is served as is.

Training needs the `flash-linear-attention` and `causal-conv1d` kernels of Qwen3.5's linear attention layers: without `causal-conv1d`, the padding-free batches raise an error, since the reference implementation can't keep the packed prompts apart (and serving without the kernels is several times slower):

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

On one L4 (24 GB): 961 steps of 32 questions, 6.8 s/step, 10.6 GB peak; one epoch takes 108 minutes. The loss goes from 0.98 to 0.59 and the accuracy on the training questions from 59% to 78%.

The trained model is on the Hub as [rycerzes/Qwen3.5-4B-Decision](https://huggingface.co/rycerzes/Qwen3.5-4B-Decision) (with your own run, use the output directory). Serve it with `transformers serve`, which needs `requests` on top of the `serving` extra:

```sh
pip install "transformers[serving]>=5.20.0" requests
transformers serve
```

```sh
curl http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "rycerzes/Qwen3.5-4B-Decision",
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
  "team": {"type": "choice", "choice": "engineering", "probabilities": {"billing": 0.004, "engineering": 0.996, "sales": 0.001}, "confidence": 0.994},
  "urgency": {"type": "score", "score": 2.98, "probabilities": {"0": 0.000, "1": 0.000, "2": 0.023, "3": 0.976}, "confidence": 0.975, "legend": {...}},
  "refund": {"type": "noul", "noul": 0.119}
}
```

(the `answers` of the response, rounded)

![Training curves](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_decision_model_metrics.png)

## 2. Evaluation

[`2_evaluate.py`](2_evaluate.py) sends the requests of two held-out sets to a running `transformers serve`, with all the questions of a state in one request:

- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), test split: 400 states (security incidents, agent traces, invoices, customer service) with 5 questions each, and soft gold probabilities from 3 annotators.
- [JevBench/jevbench](https://huggingface.co/datasets/JevBench/jevbench): 240 states with 8 to 12 questions each, of which the 1,938 whose gold is one of their options (302 have the gold `unknown`).

None of their 640 states is in the training questions, and none shares a 13-word sequence with them. Accuracy counts the questions whose most probable option is the most probable in the gold, and the Brier score is the squared distance between the probabilities of the options and the gold ones (0 is perfect, 2 the worst).

```sh
transformers serve
python examples/sft_decision_model/2_evaluate.py --model Qwen/Qwen3.5-4B --output base.jsonl
python examples/sft_decision_model/2_evaluate.py --model Qwen3.5-4B-Decision --output trained.jsonl
```

| | typed-decisions accuracy | typed-decisions Brier | JevBench accuracy | JevBench Brier |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B | 56.0% | 0.331 | 87.4% | 0.184 |
| after training | **67.3%** | **0.137** | **89.1%** | **0.141** |

By question type:

| | typed-decisions noul | choice | score | JevBench noul | choice | score |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 65.5% | 61.5% | 44.9% | 97.4% | 86.5% | 40.8% |
| after training | 74.5% | 68.2% | 61.4% | 98.4% | 89.6% | 42.9% |

The Brier score drops by 59% on typed-decisions and 23% on JevBench, where the base model already answers most `noul` and `choice` questions right. Ordinal `score` questions gain the most on typed-decisions, and remain the weakest type on JevBench.
