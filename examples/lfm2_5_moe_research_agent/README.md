# SFT and GRPO of LFM2.5-8B-A1B

[LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) is a mixture of experts with 1.5B of its 8.3B parameters active per token. This example trains it twice, both times from the released model: SFT on function calling, then GRPO as a multi-turn research agent. [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) has the same architecture, so the scripts should train it by swapping the model id (untested).

## 1. SFT on function calling

[`1_sft.py`](1_sft.py) fine-tunes all the weights on the 60k [xLAM](https://huggingface.co/papers/2406.18518) samples of [argilla/apigen-function-calling](https://huggingface.co/datasets/argilla/apigen-function-calling): a query, the tools it may use, and the one or several parallel calls that answer it, with the loss on the assistant turn only. The model is sharded at load time by transformers' `DistributedConfig`: the 32 experts are split 8 ways (expert parallelism with token dispatch), everything else is FSDP-sharded, and FSDP keeps float32 master weights.

```sh
torchrun --nproc_per_node 8 examples/lfm2_5_moe_research_agent/1_sft.py
```

![SFT loss](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/lfm2_5_moe_sft_loss.png)

One epoch on 8 H100s: 938 steps in 16 minutes, 0.80 s and 27k tokens per step, 3.7k tokens/s per GPU, 31.5 GB peak per GPU. The loss goes from 1.16 to 0.03 and the token accuracy from 90.5% to 99.3%.

The trained model is on the Hub as [trl-lib/LFM2.5-8B-A1B-SFT-function-calling](https://huggingface.co/trl-lib/LFM2.5-8B-A1B-SFT-function-calling) (with your own run, use the last checkpoint in the output directory):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.chat_template_utils import add_response_schema, parse_response


def get_weather(city: str, unit: str) -> str:
    """
    Gets the current weather in a city.

    Args:
        city: The name of the city.
        unit: The temperature unit, "celsius" or "fahrenheit".
    """


model_id = "trl-lib/LFM2.5-8B-A1B-SFT-function-calling"
tokenizer = add_response_schema(AutoTokenizer.from_pretrained(model_id))
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

messages = [{"role": "user", "content": "Is it warmer in Lisbon or in Oslo right now? Answer in Fahrenheit."}]
prompt = tokenizer.apply_chat_template(messages, tools=[get_weather], add_generation_prompt=True)["input_ids"]
output = model.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=128, do_sample=False)
print(parse_response(tokenizer, output[0, len(prompt) :].tolist(), prefix=prompt, tools=[get_weather]))
```

```
{'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_weather', 'arguments': {'city': 'Lisbon', 'unit': 'fahrenheit'}}}, {'type': 'function', 'function': {'name': 'get_weather', 'arguments': {'city': 'Oslo', 'unit': 'fahrenheit'}}}], 'content': ''}
```

## 2. GRPO: a research agent on HotpotQA

Each [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) question comes with ten Wikipedia articles: the two that hold the answer and eight distractors. The model never sees them directly. It gets the question and three tools, `search(query)` (titles and first sentences of the three best matches), `read(title)` and `submit_answer(answer)`, and has to find the first article, pull the bridge entity out of it, find the second one and submit a short answer. The reward is the exact match of the submitted answer and nothing else, so [`GRPOTrainer`] only pays for research that ends in the right answer. The environment is a plain Python class passed as `environment_factory` ([`wiki_library.py`](wiki_library.py)); its public methods become the tools.

Measure the base model on 500 held-out validation questions (all labelled `hard` by HotpotQA), greedy, with the same tool loop and TRL's tool-call parser, then train:

```sh
python examples/lfm2_5_moe_research_agent/2_evaluate.py --model LiquidAI/LFM2.5-8B-A1B --output base.jsonl
python examples/lfm2_5_moe_research_agent/3_grpo.py                                # 1 GPU, vLLM colocated
accelerate launch --num_processes 8 examples/lfm2_5_moe_research_agent/3_grpo.py    # 8 GPUs, data parallel
```

![GRPO reward](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/lfm2_5_moe_grpo_reward.png)

One H100, LoRA r=32, 4 questions x 8 episodes per step, 42 s per step (80% of it generating):

| | Exact match | Answer submitted | Episodes that `read` an article | Searches per episode |
| --- | --- | --- | --- | --- |
| LFM2.5-8B-A1B | 35.8% | 57.8% | 60.4% | 4.2 |
| after 50 steps | 42.2% | 73.0% | 62.8% | 2.5 |
| after 100 steps | 44.4% | 80.6% | 73.0% | 2.0 |
| after 150 steps | 47.6% | 88.0% | 77.0% | 2.0 |
| after 200 steps | 47.8% | 88.4% | 81.4% | 2.1 |

The base model mostly reads search snippets: it searches the same entity again and again until it runs out of turns (140 of 500 episodes, 37 after 200 steps). After training it opens the articles, follows the link to the second one, and submits. On the same 500 questions, 200 steps of training (2.4 hours) fix 87 that the base model missed and lose 27.

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

The trained agent is on the Hub as [trl-lib/LFM2.5-8B-A1B-GRPO-HotpotQA](https://huggingface.co/trl-lib/LFM2.5-8B-A1B-GRPO-HotpotQA), the last LoRA checkpoint merged into the base model so vLLM can serve it (with your own run, merge it with `AutoPeftModelForCausalLM.from_pretrained(checkpoint).merge_and_unload()`). Play an episode on the first held-out question; each episode is written to `episode.jsonl`:

```sh
python examples/lfm2_5_moe_research_agent/2_evaluate.py --model trl-lib/LFM2.5-8B-A1B-GRPO-HotpotQA --num_questions 1 --output episode.jsonl
```

```
{"model": "trl-lib/LFM2.5-8B-A1B-GRPO-HotpotQA", "num_questions": 1, "exact_match": 1.0, "submitted": 1.0, "mean_turns": 5.0, "turns_with_tool_call_block": 4, "turns_with_parsed_tool_calls": 4}
```
