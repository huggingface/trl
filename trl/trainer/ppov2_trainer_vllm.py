from collections import OrderedDict, defaultdict
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import print_rich_table
from vllm import SamplingParams, SingleGPULLM


INVALID_LOGPROB = 1.0



@dataclass
class PPOConfig(TrainingArguments):
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    deepspeed2: bool = False
    """Whether to use deepspeed to train the model"""
    deepspeed3: bool = False
    """Whether to use deepspeed to train the model"""
    debug: bool = False
    """wether to run in debug mode"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    total_episodes: Optional[int] = 1000000
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_eval_batch_size: int = 2
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    num_sample_generations: int = 10
    """the number of debugging samples generations (i.e., `generate_completions` calls) throughout training"""

    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    response_length: int = 53
    """the length of the response"""
    truncate_token: Optional[Literal["eos"]] = None
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = False
    """whether to penalize responses that do not contain `truncate_token_id`"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft model"""

    # ppo args
    num_ppo_epochs: int = 4
    """the number of epochs to train"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange: float = 0.2
    """the clip range"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""
    remove_duplicate_response_pad_tokens: bool = False


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module) -> None:
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = (
        first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    )
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(
            **kwargs,
        )
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def prepare_deepspeed2(model, train_micro_batch_size_per_gpu):
    import deepspeed
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu

    eval_ds_config = {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
        "bf16": {"enabled": True},
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    model, *_ = deepspeed.initialize(model=model, config=eval_ds_config)
    model.eval()
    print("🔥 deepspeed2 is initialized")
    return model


def prepare_deepspeed3(model, accelerator):
    import deepspeed
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    # deepspeed_states = AcceleratorState().deepspeed_plugin
    # deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0,
                    }
                )
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    print("🔥 deepspeed3 is initialized")
    return model


class PPOTrainer(Trainer):
    def __init__(
        self,
        args: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # model_init: Optional[Callable[[torch.nn.Module], None]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.policy = policy
        self.policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.callbacks = callbacks

        #########
        # calculate various batch sizes
        #########
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(args.batch_size, args.num_mini_batches)
        args.local_mini_batch_size = exact_div(args.local_batch_size, args.num_mini_batches)
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_updates = args.total_episodes // args.batch_size
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)


        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, value_model, reward_model]:
            disable_dropout(module)
        if args.truncate_token and args.truncate_token == "eos":
            args.truncate_token_id = tokenizer.eos_token_id
        self.model = PolicyAndValueWrapper(policy, value_model)
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        DEFAULT_CALLBACKS = [DefaultFlowCallback]
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        if self.callbacks is None:
            self.callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            drop_last=True,
        ) # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        #########
        ### vllm
        #########
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.response_length,
            include_stop_str_in_output=True,
        )
        if accelerator.is_main_process:
            self.llm = SingleGPULLM(model=args.base_model,
                            tensor_parallel_size=1,
                            device="cuda:7")
            self.llmp = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            print(f"🔥🔥🔥 vllm loaded")
        else:
            print("waiting for vllm to spin up...")
        accelerator.wait_for_everyone()
        if args.deepspeed2:
            self.reward_model = prepare_deepspeed2(self.reward_model, args.per_device_train_batch_size)
            ref_policy = prepare_deepspeed2(ref_policy, args.per_device_train_batch_size)
        elif args.deepspeed3:
            self.reward_model = prepare_deepspeed3(self.reward_model, accelerator)
            ref_policy = prepare_deepspeed3(ref_policy, accelerator)
        else:
            ref_policy = ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)


    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader
    
    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def push_to_hub(self, **kwargs):
        """Modified from `Trainer.save_model` to only save the policy and not the value network."""
        self.backup_model = self.model
        self.model = self.accelerator.unwrap_model(self.model).policy # save only the policy
        super().push_to_hub(**kwargs)
        self.model = self.backup_model

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Modified from `Trainer.save_model` to only save the policy and not the value network."""
        if not _internal_call: # `push_to_hub` already swaps out the self.model with policy
            self.backup_model = self.model
            self.model = self.accelerator.unwrap_model(self.model).policy # save only the policy
        if output_dir is None:
            output_dir = self.args.output_dir
        state_dict = self.accelerator.get_state_dict(self.backup_model)
        policy_state_dict = state_dict
        if self.accelerator.is_main_process:
            policy_state_dict = OrderedDict({k[len("policy."):]: v for k, v in state_dict.items() if k.startswith("policy.")})
        if self.args.should_save:
            self._save(output_dir, state_dict=policy_state_dict)
        if not _internal_call:
            self.model = self.backup_model

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())


        accelerator.print("===training policy===")
        global_step = 0
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        g_responses = torch.zeros((args.batch_size, args.response_length), device=device, dtype=torch.long)
        model.train()
        for update in range(1, args.num_updates + 1):
            global_step += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                query_responses = []
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                values = []
                scores = []
                sequence_lengths = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    g_queries_list = gather_object(queries.tolist())
                    
                    if accelerator.is_main_process:
                        print("🔥🔥🔥 Loading weights using shared memory;"
                            "we expect the generations to be completely different")
                        start_time = time.time()
                        self.llmp.load_weights(unwrapped_model.policy.named_parameters())
                        print(f"Time to load weights: {time.time() - start_time:.2f} seconds")
                        g_queries_list = [[inneritem for inneritem in item if inneritem != tokenizer.pad_token_id]  for item in g_queries_list]
                        outputs = self.llm.generate(prompt_token_ids=g_queries_list, sampling_params=self.sampling_params)
                        padded_response_token_ids = []
                        for output in outputs:
                            token_ids = output.outputs[0].token_ids
                            DUMMY_PAD_TOKEN = 0 # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
                            padded_token_ids = token_ids + [DUMMY_PAD_TOKEN] * (args.response_length - len(token_ids))
                            padded_response_token_ids.append(padded_token_ids)
                        padded_response_token_ids = torch.tensor(padded_response_token_ids, device=device)
                        g_responses[:] = padded_response_token_ids

                    broadcast(g_responses, 0)
                    local_responses = g_responses[accelerator.local_process_index * queries.shape[0]: (accelerator.local_process_index + 1) * queries.shape[0]]
                    # if args.remove_duplicate_response_pad_tokens: # NOTE: micro optimization: remove the pad to longest
                    #     local_responses = local_responses[:, :(local_responses != tokenizer.pad_token_id).sum(1).max()]
                    queries_responses = torch.cat((queries, local_responses), 1)
                    for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                        query = queries[i : i + args.local_rollout_forward_batch_size]
                        query_response = queries_responses[i : i + args.local_rollout_forward_batch_size]
                        response = query_response[:, context_length:]

                        output = forward(unwrapped_model.policy, query_response, tokenizer)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        all_logprob = F.log_softmax(logits, dim=-1)
                        logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        del output, logits, all_logprob
                        torch.cuda.empty_cache()

                        ref_output = forward(ref_policy, query_response, tokenizer)
                        ref_logits = ref_output.logits[:, context_length - 1 : -1]
                        ref_logits /= args.temperature + 1e-7
                        ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                        ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        del ref_output, ref_logits, ref_all_logprob
                        torch.cuda.empty_cache()

                        # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
                        postprocessed_response = response
                        if args.truncate_token_id:
                            postprocessed_response = truncate_response(args, tokenizer, response)

                        # Response Processing 2. run reward model on the truncated responses
                        postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                        sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                        unwrapped_value_model = accelerator.unwrap_model(model).value_model
                        full_value, _, _ = get_reward(unwrapped_value_model, query_response, tokenizer, context_length, self.rm_bias)
                        value = full_value[:, context_length - 1 : -1].squeeze(-1)
                        _, score, _ = get_reward(reward_model, postprocessed_query_response, tokenizer, context_length, self.rm_bias)

                        query_responses.append(query_response)
                        responses.append(response)
                        postprocessed_responses.append(postprocessed_response)
                        logprobs.append(logprob)
                        ref_logprobs.append(ref_logprob)
                        values.append(value)
                        sequence_lengths.append(sequence_length)
                        scores.append(score)
                query_responses = torch.cat(query_responses, 0)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                values = torch.cat(values, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
                if args.non_eos_penalty:
                    scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))
                accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                sequence_lengths_p1 = sequence_lengths + 1
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                # accelerator.print("rewards====", rewards[0])
                # accelerator.print("advantages====", advantages[0])
                # accelerator.print("values====", values[0])
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, tokenizer)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                            )
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = pg_clipfrac
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = vf_clipfrac
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
                # accelerator.print(
                #     f"ppo_epoch_idx: {ppo_epoch_idx}",
                #     f"approxkl: {approxkl_stats[:ppo_epoch_idx + 1].mean().item():.4f}",
                #     f"pg_loss: {pg_loss_stats[:ppo_epoch_idx + 1].mean().item():.4f}",
                #     f"pg_clipfrac: {pg_clipfrac_stats[:ppo_epoch_idx + 1].mean().item():.4f}",
                #     f"ratio: {ratio_stats[:ppo_epoch_idx + 1].mean().item():.4f}",
                # )
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                eps = int(global_step / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = (
                    self.accelerator.gather(mean_non_score_reward + scores.mean()).mean().item()
                )
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = global_step
                self.state.epoch = global_step / self.train_dataset_len  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del query_responses, responses, postprocessed_responses, logprobs, ref_logprobs, values, sequence_lengths, contain_eos_token, sequence_lengths_p1, response_idxs, padding_mask, padding_mask_p1, rewards, actual_start, actual_end, advantages, returns
            torch.cuda.empty_cache()

    def generate_completions(self, sampling: bool = False):
        args = self.args
        accelerator = self.accelerator
        tokenizer = self.tokenizer
        device = accelerator.device
        generation_config = SamplingParams(
            temperature=(0.01 + 1e-7),
            top_p=1.0,
            max_tokens=args.response_length,
            include_stop_str_in_output=True,
        )

        table = defaultdict(list)
        g_responses = torch.zeros((args.per_device_eval_batch_size * args.world_size, args.response_length), device=device, dtype=torch.long)
        for batch in self.eval_dataloader:
            queries = batch["input_ids"]
            name = f"trained {args.base_model}"
            with torch.no_grad():
                context_length = queries.shape[1]
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:

                    g_queries_list = gather_object(queries.tolist())
                    
                    if accelerator.is_main_process:
                        print("🔥🔥🔥 Loading weights using shared memory;"
                            "we expect the generations to be completely different")
                        start_time = time.time()
                        self.llmp.load_weights(unwrapped_model.policy.named_parameters())
                        print(f"Time to load weights: {time.time() - start_time:.2f} seconds")
                        g_queries_list = [[inneritem for inneritem in item if inneritem != tokenizer.pad_token_id]  for item in g_queries_list]
                        outputs = self.llm.generate(prompt_token_ids=g_queries_list, sampling_params=generation_config)
                        padded_response_token_ids = []
                        for output in outputs:
                            token_ids = output.outputs[0].token_ids
                            padded_token_ids = token_ids + [tokenizer.pad_token_id] * (args.response_length - len(token_ids))
                            padded_response_token_ids.append(padded_token_ids)

                        padded_response_token_ids = torch.tensor(padded_response_token_ids, device=device)

                        g_responses[:] = padded_response_token_ids

                    broadcast(g_responses, 0)
                    queries_responses = torch.cat((queries, g_responses[accelerator.local_process_index * queries.shape[0]: (accelerator.local_process_index + 1) * queries.shape[0]]), 1)

                response = queries_responses[:, context_length:]
                postprocessed_response = response
                if args.truncate_token_id:
                    postprocessed_response = truncate_response(args, self.tokenizer, response)
                table["query"].extend(gather_object(self.tokenizer.batch_decode(queries, skip_special_tokens=True)))
                table[name].extend(gather_object(self.tokenizer.batch_decode(postprocessed_response)))

                postprocessed_query_response = torch.cat((queries, postprocessed_response), 1)
                _, score, _ = get_reward(self.reward_model, postprocessed_query_response, self.tokenizer, context_length)
                table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df.iloc[0 : 0 + 5])
        if "wandb" in args.report_to:
            import wandb
            if wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})
