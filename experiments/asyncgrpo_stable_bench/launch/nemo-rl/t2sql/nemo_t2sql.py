"""Text-to-SQL (miles' River recipe, as ported for TRL in workloads/text2sql/t2sql_episode.py) on NeMo-RL.

NeMo-RL's documented custom multi-turn path (docs/guides/environments.md, examples/run_grpo_sliding_puzzle.py): a
Ray-actor `EnvironmentInterface` whose `step` sees the message log after each assistant turn and returns the next
observation, plus a driver that builds the dataset and calls NeMo-RL's own `setup` / `grpo_train` / `async_grpo_train`.
No NeMo-RL file is changed.

The episode logic is t2sql_episode.py's `run_episode`, cut at the turn boundary because NeMo-RL owns the generation
loop. Every helper (prompt, SQL tool, observation framing, reward) is imported from t2sql_episode.py itself, so the
prompt ids, the tool, the truncation, the turn budget and the reward are the TRL port's code, not a copy:
- prompt: `opening_prompt_ids` (thinking on, the template default);
- per-turn budget: NeMo-RL clamps each turn to min(max_new_tokens, max_model_len - input) = min(3072, 29696 - len);
- observations: the exact text `_observation_ids` encodes (eos-close + "\n" + chat-template framing); NeMo-RL tokenizes
  the observation with `tokenizer(text, add_special_tokens=False)`, which gives the same ids;
- reward: `reward_func` on the terminal step, 0 on the others (NeMo-RL sums per-turn rewards).
One forced difference: NeMo-RL ends a rollout after any turn that hits its token limit (`_response_truncated`), so a
3,072-token turn that still contains a tool call is scored as unanswered (-1) instead of getting its tool result. The
base model hits the per-turn cap on 0% of turns (workloads/text2sql/README.md item 5), so this should be rare; it is counted in
episodes.jsonl (`status == "length_cut"`).
"""

import asyncio
import json
import os
import pprint
import sys
import time
import types
from collections import namedtuple

import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


# t2sql_episode.py imports TRL's rollout classes for its TRL loop; NeMo-RL's container has no TRL, and only the episode
# helpers are used here, so stand-ins for those three names let the module import unchanged.
def _install_trl_stub():
    if "trl.experimental.async_grpo.async_rollout_worker" in sys.modules:
        return
    stub = types.ModuleType("trl.experimental.async_grpo.async_rollout_worker")
    stub.AsyncRolloutLoop = object
    stub.Rollout = None
    stub.TurnRecord = namedtuple("TurnRecord", "prompt_ids output_ids logprobs")
    for name in ("trl", "trl.experimental", "trl.experimental.async_grpo"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[stub.__name__] = stub


_install_trl_stub()
import t2sql_episode as ep  # noqa: E402
from text_to_sql_recipe import parsing, prompts, reward  # noqa: E402

from nemo_rl.algorithms.grpo import MasterConfig, async_grpo_train, grpo_train, setup  # noqa: E402
from nemo_rl.algorithms.utils import get_tokenizer, set_seed  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.distributed.virtual_cluster import init_ray  # noqa: E402
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn  # noqa: E402
from nemo_rl.models.generation import configure_generation_config  # noqa: E402
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers  # noqa: E402
from nemo_rl.utils.logger import get_next_experiment_dir  # noqa: E402


TASK_NAME = "text2sql"


@ray.remote(max_concurrency=4096)  # async actor: steps of different episodes run concurrently on its event loop
class Text2SQLEnv(EnvironmentInterface):
    def __init__(self, model_name: str, episodes_file: str, stop_token_ids: list[int]):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stop_token_ids = set(stop_token_ids)
        self.episodes = open(episodes_file, "a", buffering=1)

    async def step(self, message_log_batch, metadata):
        results = await asyncio.gather(*(self._step_one(log, meta) for log, meta in zip(message_log_batch, metadata)))
        observations, next_meta, rewards, terminateds = zip(*results)
        return EnvironmentReturn(
            observations=list(observations),
            metadata=list(next_meta),
            next_stop_strings=[None] * len(results),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=[None] * len(results),
        )

    async def _step_one(self, log, meta):
        """One turn of t2sql_episode.run_episode, from `episode["turns"].append(text)` to the next observation."""
        task, episode, turn = meta["task"], meta["episode"], meta["turn"]
        if turn == 1:
            meta["t0"] = time.time()
        # NeMo-RL hands the env only role/content (calculate_rewards -> get_keys_from_message_log), so the turn's
        # length is re-derived from its text: content is the sampled ids decoded with skip_special_tokens (vLLM's
        # `text`); a turn that used its whole budget is the length-cut one, otherwise it ended on eos (+1 token)
        text = log[-1]["content"]
        n = len(self.tokenizer.encode(text, add_special_tokens=False))
        max_new_tokens = min(ep.MAX_TOKENS_PER_TURN, ep.MAX_SEQ_LEN - meta["total"])
        finish_reason = "length" if n >= max_new_tokens - 1 else "stop"
        n_sampled = n if finish_reason == "length" else n + 1
        total = meta["total"] + n_sampled
        meta["sampled_tokens"] += n_sampled
        meta["finish_reasons"].append(finish_reason)

        episode["turns"].append(text)
        message = {"role": "assistant", "content": ep._visible_text(text)}
        tools_allowed = turn <= ep.MAX_TOOL_TURNS
        reward.observe_tool_calls(episode["tool_protocol_events"], message, turn=turn, tools_allowed=tools_allowed)
        if finish_reason == "length" and max_new_tokens < ep.MAX_TOKENS_PER_TURN:
            return await self._finish(meta, total, "context_cut")
        if not tools_allowed:
            episode["final_text"] = message["content"].strip()
            return await self._finish(meta, total, "completed")

        call = next((c for c in parsing.parse_tool_calls(message) if c.name == prompts.SQL_TOOL_NAME), None)
        if call is not None and finish_reason == "length":
            # NeMo-RL stops the rollout after a length-truncated turn; score it now (see the module docstring)
            return await self._finish(meta, total, "length_cut")
        if call is not None:
            episode["tool_calls"] += 1
            observation = await ep._sql_observation(task, call.arguments, turns_left=ep.MAX_TOOL_TURNS - turn)
            if observation is None:
                episode["unsafe"] = True
                return await self._finish(meta, total, "completed")
        elif parsing.has_tool_call_intent(message):
            observation = parsing.MALFORMED_ACTION_PROMPT
        else:
            episode["final_text"] = message["content"].strip()
            return await self._finish(meta, total, "completed")
        if finish_reason == "length":
            # same as above: NeMo-RL ends the rollout here, so this turn is the last one
            return await self._finish(meta, total, "length_cut")

        messages = [{"role": "user", "content": observation}]
        if turn == ep.MAX_TOOL_TURNS:
            messages.append({"role": "user", "content": prompts.SQL_LIMIT_PROMPT})
        stop = "" if finish_reason == "stop" else self.tokenizer.eos_token
        framing = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        observation_text = stop + "\n" + framing  # = the text ep._observation_ids encodes
        observation_ids = self.tokenizer.encode(observation_text, add_special_tokens=False)
        if total + len(observation_ids) >= ep.MAX_SEQ_LEN:
            return await self._finish(meta, total, "truncated")
        meta["turn"] = turn + 1
        meta["total"] = total + len(observation_ids)
        return {"role": "user", "content": observation_text}, meta, 0.0, False

    async def _finish(self, meta, total, status):
        scores = await ep.reward_func(meta["task"], meta["episode"], evaluation=False)
        record = {
            "id": str(meta["task"]["id"]),
            "evaluation": False,
            "status": status,
            "turns": meta["turn"],
            "finish_reasons": meta["finish_reasons"],
            "prompt_tokens": meta["prompt_tokens"],
            "total_tokens": total,
            "sampled_tokens": meta["sampled_tokens"],
            **scores,
            "start": meta.get("t0"),
            "end": time.time(),
        }
        self.episodes.write(json.dumps(record) + "\n")
        return {"role": "user", "content": ""}, None, float(scores["score"]), True

    def shutdown(self):
        self.episodes.close()

    def global_post_process_and_metrics(self, batch: BatchedDataDict):
        rewards = batch["total_reward"]
        return batch, {"reward_mean": rewards.mean().item(), "correct_frac": (rewards >= 0.5).float().mean().item()}


class Text2SQLDataset(Dataset):
    """convert.py's train split (2,033 tasks), each row a NeMo-RL DatumSpec with the TRL port's prompt ids."""

    def __init__(self, path: str, tokenizer):
        self.rows = []
        for i, line in enumerate(open(path)):
            task = json.loads(json.loads(line)["task"])
            ids = ep.opening_prompt_ids(tokenizer, task)
            self.rows.append((i, task, ids, tokenizer.decode(ids)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        i, task, ids, text = self.rows[idx]
        episode = {"turns": [], "final_text": "", "unsafe": False, "tool_calls": 0, "tool_protocol_events": []}
        meta = {"task": task, "episode": episode, "turn": 1, "prompt_tokens": len(ids), "total": len(ids), "sampled_tokens": 0,
                "finish_reasons": []}
        return {
            "message_log": [{"role": "user", "content": text, "token_ids": torch.tensor(ids, dtype=torch.long)}],
            "length": len(ids),
            "extra_env_info": meta,
            "loss_multiplier": 1.0,
            "idx": i,
            "task_name": TASK_NAME,
            "stop_strings": None,
        }


def main():
    register_omegaconf_resolvers()
    config = load_config(sys.argv[1])
    if len(sys.argv) > 2:
        config = parse_hydra_overrides(config, sys.argv[2:])
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    pprint.pprint(config)
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])

    init_ray()
    set_seed(config.grpo.seed)
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(config.policy["generation"], tokenizer)

    dataset = Text2SQLDataset(os.environ["T2SQL_TRAIN"], tokenizer)
    stop_ids = sorted({tokenizer.eos_token_id, *(config.policy["generation"].get("stop_token_ids") or [])})
    print(f"train rows {len(dataset)}, stop token ids {stop_ids}", flush=True)
    env = Text2SQLEnv.options(num_gpus=0).remote(config.policy["model_name"], os.environ["T2SQL_EPISODES_FILE"], stop_ids)
    task_to_env = {TASK_NAME: env}

    policy, policy_generation, _gym, cluster, dataloader, val_dataloader, loss_fn, logger, checkpointer, grpo_state, \
        master_config, *_ = setup(config, tokenizer, dataset, None)
    train = async_grpo_train if config.grpo.async_grpo.enabled else grpo_train
    kwargs = {}
    if config.grpo.async_grpo.enabled:
        kwargs = {"max_trajectory_age_steps": config.grpo.async_grpo.max_trajectory_age_steps}
    train(policy, policy_generation, dataloader, val_dataloader, tokenizer, loss_fn, task_to_env, task_to_env, logger,
          checkpointer, grpo_state, master_config, **kwargs)


if __name__ == "__main__":
    main()
