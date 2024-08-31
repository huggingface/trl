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
    logging,
)

from ..models.utils import unwrap_model_for_generation
from .judges import BasePairwiseJudge
from .online_dpo_config import OnlineDPOConfig
from .utils import (
    DPODataCollatorWithPadding,
    empty_cache,
    get_reward,
    prepare_deepspeed,
    trl_sanitze_kwargs_for_tagging,
    truncate_right,
)


if is_apex_available():
    from apex import amp


if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)


class OnlineDPOTrainer(Trainer):
    r"""
    Initialize OnlineDPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        reward_model (`transformers.PreTrainedModel`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        judge (`BasePairwiseJudge`):
            The judge to use for pairwise comparison of model completions.
        args (`OnlineDPOConfig`):
            The online DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "online-dpo"]

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
        elif reward_model is None and judge is not None:
            raise NotImplementedError("Using `judge` is not yet supported.")

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

        self.stats = {
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "objective/rlhf_reward": [],
            "objective/scores": [],
            "objective/scores_margin": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "val/contain_eos_token": [],
        }

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

    # Same as Trainer.get_eval_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        # Sample 2 completations per prompt of size `max_new_tokens` from the model
        inputs = self._prepare_inputs(inputs)
        generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            min_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            use_cache=False if self.args.gradient_checkpointing else True,
        )
        num_examples, context_length = inputs["prompt_input_ids"].shape
        prompt_ids = inputs["prompt_input_ids"].repeat(2, 1)
        prompt_mask = inputs["prompt_attention_mask"].repeat(2, 1)
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config,
            )
        del inputs

        completion_ids = output[:, context_length:]
        completion_ids, completion_mask = truncate_right(
            completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)
        # There is 1 offset, because the model predict the next token
        logits = output.logits[:, context_length - 1 : -1]
        # Turn logits into logprobs
        all_logprobs = F.log_softmax(logits, dim=-1)
        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(all_logprobs, completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        del output, logits, all_logprobs  # free memory
        empty_cache()

        # Same for the reference model
        with torch.no_grad():
            ref_output = self.ref_model(prompt_completion_ids, attention_mask=prompt_completion_mask)
        ref_logits = ref_output.logits[:, context_length - 1 : -1]
        ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
        ref_logprobs = torch.take_along_dim(ref_all_logprobs, completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        del ref_output, ref_logits, ref_all_logprobs  # free memory
        empty_cache()

        # Get the reward from the reward model
        with torch.no_grad():
            _, scores, _ = get_reward(
                self.reward_model, prompt_completion_ids, self.tokenizer.pad_token_id, context_length
            )

        # Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a low (fixed) score
        contain_eos_token = torch.any(completion_ids == self.tokenizer.eos_token_id, dim=-1)
        if self.args.missing_eos_penalty is not None:
            scores[~contain_eos_token] -= self.args.missing_eos_penalty

        # Replace the logprobs of the padding tokens by 1.0
        padding_mask = ~completion_mask.bool()
        logprobs = logprobs.masked_fill(padding_mask, 1.0)
        ref_logprobs = ref_logprobs.masked_fill(padding_mask, 1.0)

        # Split the scores in 2 (the prompts of the first half are the same as the second half)
        first_half, second_half = scores.split(num_examples)

        # Get the indices of the chosen and rejected examples
        num_examples_range = torch.arange(num_examples, device=scores.device)
        mask = first_half >= second_half
        chosen_indices = num_examples_range + (~mask * num_examples)
        rejected_indices = num_examples_range + (mask * num_examples)

        # Build tensor so that the first half is the chosen examples and the second half the rejected examples
        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

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

        # Log everything
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        self.stats["logps/chosen"].append(self.accelerator.gather(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather(rejected_logprobs_sum).mean().item())
        self.stats["objective/scores"].append(self.accelerator.gather(scores.mean()).mean().item())
        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather(mean_kl).mean().item())
        non_score_reward = (-self.args.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(self.accelerator.gather(mean_non_score_reward).mean().item())
        rlhf_reward = scores + non_score_reward
        self.stats["objective/rlhf_reward"].append(self.accelerator.gather(rlhf_reward).mean().item())
        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather(mean_entropy).mean().item())
        scores_margin = scores[chosen_indices] - scores[rejected_indices]
        self.stats["objective/scores_margin"].append(self.accelerator.gather(scores_margin.mean()).mean().item())
        chosen_rewards = self.args.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.args.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

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

    # Same as Trainer.evaluate but log our metrics
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                logs[key] = sum(val) / len(val)
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(
        self,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        **kwargs,
    ) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "online-dpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        Unlike the parent class, we don't use the `token` argument to mitigate security risks.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
