# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch

from ...data_utils import is_conversational
from ...extras.profiling import profiling_context
from ...import_utils import is_vllm_available


if is_vllm_available():
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams


def _build_colocate_sampling_params(
    trainer,
    overrides: dict[str, Any] | None = None,
    *,
    logprobs: bool = True,
) -> "SamplingParams":
    if trainer.structured_outputs_regex:
        structured_outputs = StructuredOutputsParams(regex=trainer.structured_outputs_regex)
    else:
        structured_outputs = None

    generation_kwargs: dict[str, Any] = {
        "n": 1,
        "temperature": trainer.temperature,
        "top_k": trainer.top_k,
        "min_p": 0.0 if trainer.min_p is None else trainer.min_p,
        "max_tokens": trainer.max_completion_length,
        "structured_outputs": structured_outputs,
    }
    if trainer.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = trainer.repetition_penalty
    if trainer.top_p is not None:
        generation_kwargs["top_p"] = trainer.top_p
    if logprobs:
        generation_kwargs["logprobs"] = 0

    if trainer.args.generation_kwargs is not None:
        generation_kwargs.update(trainer.args.generation_kwargs)

    if overrides is not None:
        generation_kwargs.update(overrides)

    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    sampling_params = SamplingParams(**generation_kwargs)
    if sampling_params.n != 1:
        raise ValueError("generate_rollout_completions expects n=1 when using colocated vLLM.")
    return sampling_params


def generate_rollout_completions(
    trainer,
    prompts: list[str],
    *,
    generation_overrides: dict[str, Any] | None = None,
    as_chat: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Generate completions for custom rollouts when vLLM is running in colocate mode.

    Returns one result per prompt, containing prompt and completion token ids along with per-token log probabilities
    and the generated text.
    """

    if not prompts:
        return []

    if not trainer.use_vllm or trainer.vllm_mode != "colocate":
        raise RuntimeError("Custom rollouts require vLLM in colocate mode to call generate_rollout_completions.")

    sampling_params = _build_colocate_sampling_params(trainer, generation_overrides)
    prompts_for_generation = prompts
    original_size = len(prompts)

    if trainer.vllm_tensor_parallel_size > 1:
        gathered_prompts = [None for _ in range(trainer.vllm_tensor_parallel_size)]
        torch.distributed.all_gather_object(gathered_prompts, prompts, group=trainer.tp_group)
        prompts_for_generation = [prompt for group_prompts in gathered_prompts for prompt in group_prompts]

    if as_chat is None:
        as_chat = prompts_for_generation and is_conversational({"prompt": prompts_for_generation[0]})

    if trainer.args.vllm_enable_sleep_mode:
        trainer.llm.wake_up(tags=["kv_cache"])
        # Work around for https://github.com/vllm-project/vllm/issues/29341
        trainer.llm.collective_rpc("reload_weights")

    with profiling_context(trainer, "vLLM.generate_rollout"):
        if as_chat:
            vllm_outputs = trainer.llm.chat(prompts_for_generation, sampling_params=sampling_params, use_tqdm=False)
        else:
            vllm_outputs = trainer.llm.generate(
                prompts_for_generation, sampling_params=sampling_params, use_tqdm=False
            )

    results: list[dict[str, Any]] = []
    for request in vllm_outputs:
        if not request.outputs:
            results.append({"prompt_ids": request.prompt_token_ids, "completion_ids": [], "logprobs": [], "text": ""})
            continue
        sequence = request.outputs[0]
        logprobs = [next(iter(token_logprob.values())).logprob for token_logprob in sequence.logprobs]
        results.append(
            {
                "prompt_ids": request.prompt_token_ids,
                "completion_ids": sequence.token_ids,
                "logprobs": logprobs,
                "text": sequence.text,
            }
        )

    if trainer.vllm_tensor_parallel_size > 1:
        local_rank_in_group = torch.distributed.get_rank(group=trainer.tp_group)
        tp_slice = slice(local_rank_in_group * original_size, (local_rank_in_group + 1) * original_size)
        results = results[tp_slice]

    if trainer.args.vllm_enable_sleep_mode:
        trainer.llm.sleep(level=2)

    return results
