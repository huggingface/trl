# GRPO: LFM2.5-8B-A1B as a research agent on HotpotQA

Each [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) question comes with ten Wikipedia articles: the two that hold the answer and eight distractors. The model never sees them directly. It gets the question and three tools, `search(query)` (titles and first sentences of the three best matches), `read(title)` and `submit_answer(answer)`, and has to find the first article, pull the bridge entity out of it, find the second one and submit a short answer. The reward is the exact match of the submitted answer and nothing else, so [`GRPOTrainer`] only pays for research that ends in the right answer. The environment is a plain Python class passed as `environment_factory` ([`wiki_library.py`](wiki_library.py)); its public methods become the tools.

[LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) is a mixture of experts with 1.5B active parameters, which keeps the many short generation turns of an episode cheap.

## Run

```sh
python examples/sft_grpo_lfm2_5_moe/3_grpo.py                                # 1 GPU, vLLM colocated
accelerate launch --num_processes 8 examples/sft_grpo_lfm2_5_moe/3_grpo.py    # 8 GPUs, data parallel
```

Evaluate a model on 500 held-out validation questions (all labelled `hard` by HotpotQA), greedy, with the same tool loop and TRL's tool-call parser:

```sh
python examples/sft_grpo_lfm2_5_moe/2_evaluate.py --model LiquidAI/LFM2.5-8B-A1B --output base.jsonl
```

## Results

One H100, LoRA r=32, 4 questions x 8 episodes per step, ~49 s per step (81% of it generating).

| | Exact match | Answer submitted | Episodes that `read` an article | Searches per episode |
| --- | --- | --- | --- | --- |
| LFM2.5-8B-A1B | 35.8% | 57.8% | 60.4% | 4.2 |
| after 50 steps | 42.2% | 73.0% | 62.8% | 2.5 |
| after 100 steps | 44.4% | 80.6% | 73.0% | 2.0 |
| after 150 steps | 47.6% | 88.0% | 77.0% | 2.0 |
| after 200 steps | 47.8% | 88.4% | 81.4% | 2.1 |

The base model mostly reads search snippets: it searches the same entity again and again until it runs out of turns (140 of 500 episodes, 37 after 200 steps). After training it opens the articles, follows the link to the second one, and submits. On the same 500 questions, 200 steps of training (2.4 hours) fix 87 that the base model missed and lose 27.

## A held-out question, before and after

**Who is the current drummer of the band who did the song "What Lovers Do"?** (answer: Matt Flynn)

Base model, out of turns without an answer:

```
search(query='What Lovers Do')     -> What Lovers Do: "What Lovers Do" is a song by American pop rock band Maroon 5 ...
search(query='Maroon 5 drummer')   -> Maroon 5: Maroon 5 is an American pop rock band that originated in Los Angeles, California. ...
search(query='Maroon 5')           -> Maroon 5: Maroon 5 is an American pop rock band that originated in Los Angeles, California. ...
... 7 more searches for Maroon 5 ...
```

After 100 steps:

```
search(query='What Lovers Do')     -> What Lovers Do: "What Lovers Do" is a song by American pop rock band Maroon 5 ...
read(title='What Lovers Do')       -> "What Lovers Do" is a song by American pop rock band Maroon 5 featuring American R&B singer Sza. ...
search(query='Maroon 5')           -> Maroon 5: Maroon 5 is an American pop rock band that originated in Los Angeles, California. ...
read(title='Maroon 5')             -> Maroon 5 is an American pop rock band ... It currently consists of ... drummer Matt Flynn and keyboardist PJ Morton.
submit_answer(answer='Matt Flynn') -> Answer submitted.
```
