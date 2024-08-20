import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from accelerate import PartialState
from datasets import Dataset
from packaging import version
from torch.utils.data import DataLoader, IterableDataset
from transformers import DataCollator, GenerationConfig, PreTrainedTokenizerBase, Trainer, TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)

from ..models.utils import unwrap_model_for_generation
from .judges import BasePairwiseJudge
from .online_dpo_config import OnlineDPOConfig
from .utils import (
    DPODataCollatorWithPadding,
    batch_generation,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    truncate_response,
)


if is_apex_available():
    from apex import amp


if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)


class ODPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[OnlineDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.ref_model = ref_model

        if reward_model is not None and judge is not None:
            warnings.warn(
                "Both `reward_model` and `judge` are provided. Please choose provide only one of them. "
                "Ignoring `judge` and using `reward_model`."
            )
        elif reward_model is None and judge is None:
            raise ValueError("Either `reward_model` or `judge` must be provided.")

        self.reward_model = reward_model
        self.judge = judge

        if args is None:
            raise ValueError("`args` must be provided.")

        # Check that the tokenizer is provided
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided.")

        # We don't optimize the reward model model nor the ref model, so we can set them to eval mode
        self.ref_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # Define the collator is not provided
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id)

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # Tokenize the dataset
            fn_kwargs = {"is_encoder_decoder": model.config.is_encoder_decoder, "tokenizer": tokenizer}
            train_dataset = train_dataset.map(self.tokenize_row, fn_kwargs=fn_kwargs, num_proc=args.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, fn_kwargs=fn_kwargs, num_proc=args.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Placed after the super().__init__ because we need self.is_deepspeed_enabled and self.accelerator
        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.ref_model = prepare_deepspeed(self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16)
        else:
            self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)

    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
        """Tokenize a single row from a DPO specific dataset."""
        if not is_encoder_decoder:
            batch = tokenizer(feature["prompt"], add_special_tokens=False)
            # Add BOS token to head of prompt. Avoid adding if it's already there
            if tokenizer.bos_token_id is not None:
                prompt_len_input_ids = len(batch["input_ids"])
                if prompt_len_input_ids == 0 or tokenizer.bos_token_id != batch["input_ids"][0]:
                    batch["input_ids"] = [tokenizer.bos_token_id] + batch["input_ids"]
                    batch["attention_mask"] = [1] + batch["attention_mask"]
        else:
            batch = tokenizer(feature["prompt"], add_special_tokens=True)
        batch = {f"prompt_{key}": value for key, value in batch.items()}
        return batch

    # Same as Trainer.get_train_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        inputs = self._prepare_inputs(inputs)

        # Generate two completions
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            min_new_tokens=self.args.response_length,
            temperature=(self.args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        repeated_prompts = inputs["prompt_input_ids"].repeat(2, 1)
        with torch.no_grad():
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                query_responses, logits = batch_generation(
                    model=unwrapped_model,
                    queries=repeated_prompts,  # generate 2 completions per prompt
                    local_rollout_forward_batch_size=self.args.per_device_train_batch_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    generation_config=generation_config,
                )
            num_examples, context_length = inputs["prompt_input_ids"].shape
            responses = query_responses[:, context_length:]  # responses.shape[1] == self.args.response_length
            # Turn logits into logprobs
            all_logprobs = F.log_softmax(logits, dim=-1)  # (batch_size, response_length, vocab_size)
            # Take the response tokens logprob (batch_size, response_length)
            logprobs = torch.take_along_dim(all_logprobs, responses.unsqueeze(-1), dim=2).squeeze(-1)

            # Same for the reference model
            ref_output = forward(self.ref_model, query_responses, pad_token_id=self.tokenizer.pad_token_id)
            # There is 1 offset, because the model predict the next token
            ref_logits = ref_output.logits[:, context_length - 1 : -1] / generation_config.temperature
            # Turn logits into logprobs
            ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
            # Take the response tokens logprob
            ref_logprobs = torch.take_along_dim(ref_all_logprob, responses.unsqueeze(-1), dim=2).squeeze(-1)

            # Truncate response after the first occurrence of `stop_token_id`.
            processed_responses = truncate_response(
                self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, responses
            )
            # Reponses now look like: [123, 234, 345, EOS, PAD, PAD, ...]

            # Run reward model on the truncated responses
            processed_query_responses = torch.hstack((repeated_prompts, processed_responses))
            _, scores, _ = get_reward(
                self.reward_model, processed_query_responses, self.tokenizer.pad_token_id, context_length
            )

            # Filter response. Ensure that the sample contains stop_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_eos_token = torch.any(processed_responses == self.tokenizer.eos_token_id, dim=-1)
            if self.args.missing_eos_penalty is not None:
                scores = torch.where(contain_eos_token, scores, self.args.missing_eos_penalty)

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            # response_idxs looks like tensor([[ 0,  1,  2,  3], [ 0,  1,  2,  3]])
            sequence_lengths = first_true_indices(processed_responses == self.tokenizer.pad_token_id)
            # The seq_len-th token is the EOS: [234, 345, EOS, PAD, PAD, ...] -> sequence_length = 2
            padding_mask = response_idxs > (sequence_lengths.unsqueeze(1) - 1)
            # With the above example, logprobs must look like [0.1, 0.2, 0.3, 1.0, 1.0, ...]
            logprobs = logprobs.masked_fill(padding_mask, 1.0)
            ref_logprobs = ref_logprobs.masked_fill(padding_mask, 1.0)

            # Compute the rewards
            kl = logprobs - ref_logprobs
            non_score_reward = (-self.args.beta * kl).sum(1)
            rlhf_reward = scores + non_score_reward
            mean_entropy = (-logprobs).sum(1).mean()

            # Split the scores in 2
            first_half, second_half = scores.split(num_examples)

            # Get the indices of the chosen and rejected examples
            num_examples_range = torch.arange(num_examples, device=scores.device)
            mask = first_half >= second_half
            chosen_indices = num_examples_range + (~mask * num_examples)
            rejected_indices = num_examples_range + (mask * num_examples)
            scores_margin = scores[chosen_indices] - scores[rejected_indices]

        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)

        cr_responses = responses[cr_indices]
        cr_query_responses = query_responses[cr_indices]
        cr_output = forward(model, cr_query_responses, self.tokenizer.pad_token_id)
        cr_logits = cr_output.logits
        cr_logits = cr_logits[:, context_length - 1 : -1] / generation_config.temperature
        cr_all_log_probs = F.log_softmax(cr_logits, dim=-1)
        cr_logprobs = torch.take_along_dim(cr_all_log_probs, cr_responses.unsqueeze(-1), dim=2)
        cr_logprobs = cr_logprobs.squeeze(-1)
        cr_logprobs = cr_logprobs.masked_fill(padding_mask[cr_indices], 1.0)

        cr_logprobs_sum = (cr_logprobs * ~padding_mask[cr_indices]).sum(1)
        cr_ref_logprobs_sum = (ref_logprobs[cr_indices] * ~padding_mask[cr_indices]).sum(1)

        # Split the chosen and rejected examples
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, num_examples)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, num_examples)
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.args.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.args.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.loss_type}")

        loss = losses.mean()
        del inputs

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
