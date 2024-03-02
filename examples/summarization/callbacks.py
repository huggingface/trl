import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import accelerate
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, TrainerCallback

import wandb
from trl.trainer.utils import pad_to_length


@dataclass
class PromptAndTextCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_prompt_length: Optional[int] = None
    max_length: Optional[int] = None
    prompt_field: str = "prompt"
    target_field: str = "label"
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [feat[self.prompt_field] for feat in features]
        texts = [feat[self.prompt_field] + " " + feat[self.target_field] for feat in features]

        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        tokenized_batch = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.max_prompt_length,
            return_tensors=self.return_tensors,
        )
        tokenized_batch["prompt"] = prompts

        self.tokenizer.padding_side = original_side

        tokenized_texts = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )

        text_labels = tokenized_texts["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            text_labels[text_labels == self.tokenizer.pad_token_id] = -100

        tokenized_batch.update(
            {
                "text_input_ids": tokenized_texts["input_ids"],
                "text_attention_mask": tokenized_texts["attention_mask"],
                "text_labels": text_labels,
            }
        )

        return tokenized_batch


class GoldModelRewardCallback(TrainerCallback):
    def __init__(
        self,
        args,
        gold_model,
        gold_eval_dataset,
        tokenizer,
        accelerator,
        max_length,
        max_prompt_length,
        prompt_field,
        target_field,
        gold_load_and_unload=False,
        log_n_samples_during_eval=0,
        generation_config=None,
    ):
        self.max_length = max_length
        self.log_n_samples_during_eval = log_n_samples_during_eval
        self.generation_config = generation_config

        # data_collator = DataCollatorWithPadding(tokenizer)
        data_collator = PromptAndTextCollator(
            tokenizer,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
            prompt_field=prompt_field,
            target_field=target_field,
        )
        dataloader_params = {
            "batch_size": args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        dataloader = DataLoader(gold_eval_dataset, **dataloader_params)
        self.dataloader = accelerator.prepare(dataloader)
        self.accelerator = accelerator
        self.completed_step = -1
        self.gold_model = gold_model
        self.gold_load_and_unload = gold_load_and_unload
        # keep model on gpu the whole time
        if not self.gold_load_and_unload:
            self.gold_model = self.accelerator.prepare(self.gold_model)

    def on_evaluate(self, args, state, control, model, tokenizer, metrics, **kwargs):
        samples_to_log = []
        gold_reward_sum = 0.0
        nll_sum = 0.0
        total_samples = 0
        sample_length_sum = 0.0

        # load model onto gpu for inference then unload
        if self.gold_load_and_unload:
            self.gold_model = self.accelerator.prepare(self.gold_model)

        if state.global_step == self.completed_step:
            return

        for inputs in tqdm(
            self.dataloader, desc="Gold Eval", dynamic_ncols=True, disable=not state.is_local_process_zero
        ):
            # get loss over true continuation i.e. ppl on dataset
            with torch.no_grad():
                nll_loss = model(
                    input_ids=inputs["text_input_ids"],
                    attention_mask=inputs["text_attention_mask"],
                    labels=inputs["text_labels"],
                ).loss

            nll_loss = self.accelerator.gather_for_metrics(nll_loss)

            # generate from model
            policy_output_decoded, ref_output_decoded, policy_output_ids = self.get_batch_samples(
                model,
                tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
                return_ids=True,
            )

            # gold reward
            policy_output_attention_mask = (policy_output_ids != tokenizer.pad_token_id).to(torch.int64)
            with torch.no_grad():
                gold_rewards = self.gold_model(
                    input_ids=policy_output_ids, attention_mask=policy_output_attention_mask
                )[0]

            gold_rewards = self.accelerator.gather_for_metrics(gold_rewards)

            if state.is_local_process_zero:
                nll_sum += nll_loss.sum().item()
                gold_reward_sum += gold_rewards.sum().item()
                total_samples += gold_rewards.size(0)
                sample_length_sum += policy_output_attention_mask.sum().item()

                # Sample and save to game log if requested (for one batch to save time)
                for i, (prompt, pol, ref) in enumerate(
                    zip(inputs["prompt"], policy_output_decoded, ref_output_decoded)
                ):
                    if len(samples_to_log) < self.log_n_samples_during_eval:
                        samples_to_log.append([prompt, pol[len(prompt) :], ref[len(prompt) :]])
                    else:
                        break

        if self.gold_load_and_unload:
            self.gold_model = self.gold_model.to("cpu")
            torch.cuda.empty_cache()

        if state.is_world_process_zero:
            gold_log = {
                "eval/gold_rewards_mean": gold_reward_sum / total_samples,
                "eval/perplexity": math.exp(nll_sum / total_samples),
                "eval/gold_sample_length": sample_length_sum / total_samples,
            }
            for key, value in gold_log.items():
                print(f"{key}: {value}")
            if state.epoch:
                gold_log["epoch"] = round(state.epoch, 2)
                gold_log["step"] = state.global_step
            if samples_to_log:
                gold_log["gold_log"] = (
                    wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=samples_to_log,
                    ),
                )
            wandb.log(gold_log)

        self.completed_step = state.global_step

    def get_batch_samples(self, model, tokenizer, input_ids, attention_mask, return_ids=False) -> Tuple[str, str]:
        """Reduce inputs to unseen prompts, and maximum batch size if necessary
        Generate samples from the model and reference model for the given batch of inputs."""
        policy_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )

        # if self.ref_model is None:
        with self.accelerator.unwrap_model(model).disable_adapter():
            reference_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
            )
        # else:
        #     reference_output = self.ref_model.generate(
        #         **inputs,
        #         generation_config=self.generation_config,
        #     )

        policy_output = pad_to_length(policy_output, self.max_length, tokenizer.pad_token_id)
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, tokenizer.pad_token_id)
        reference_output_decoded = tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        if return_ids:
            return policy_output_decoded, reference_output_decoded, policy_output
        else:
            return policy_output_decoded, reference_output_decoded


class PerplexityGenCallback(TrainerCallback):
    """Like GoldModelReward in that you generate and get ppl on dataset

    But you don't run eval with the gold model
    Useful when gold model is very larger and you want to run inference later
    """

    def __init__(
        self,
        args,
        dataset,
        tokenizer,
        accelerator,
        max_length,
        max_prompt_length,
        prompt_field,
        target_field,
        log_n_samples_during_eval=0,
        generation_config=None,
        hub_name="tmp",
    ):
        self.max_length = max_length
        self.log_n_samples_during_eval = log_n_samples_during_eval
        self.generation_config = generation_config

        # data_collator = DataCollatorWithPadding(tokenizer)
        data_collator = PromptAndTextCollator(
            tokenizer,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
            prompt_field=prompt_field,
            target_field=target_field,
        )
        dataloader_params = {
            "batch_size": args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        dataloader = DataLoader(dataset, **dataloader_params)
        self.dataloader = accelerator.prepare(dataloader)
        self.accelerator = accelerator
        self.completed_step = -1
        self.hub_name = hub_name

    def on_evaluate(self, args, state, control, model, tokenizer, metrics, **kwargs):
        all_generations = []
        all_prompts = []
        nll_sum = 0.0
        total_samples = 0
        sample_length_sum = 0.0

        if state.global_step == self.completed_step:
            return

        for inputs in tqdm(
            self.dataloader, desc="PPL and Gen Eval", dynamic_ncols=True, disable=not state.is_local_process_zero
        ):
            # get loss over true continuation i.e. ppl on dataset
            with torch.no_grad():
                nll_loss = model(
                    input_ids=inputs["text_input_ids"],
                    attention_mask=inputs["text_attention_mask"],
                    labels=inputs["text_labels"],
                ).loss

            # generate from model
            policy_output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config,
            )
            policy_output_ids = pad_to_length(policy_output_ids, self.max_length, tokenizer.pad_token_id)

            policy_output_attention_mask = (policy_output_ids != tokenizer.pad_token_id).to(torch.int64)
            generation_sizes = policy_output_attention_mask.sum(dim=1)

            (nll_loss, generation_ids, generation_sizes) = self.accelerator.gather_for_metrics(
                (nll_loss, policy_output_ids, generation_sizes)
            )

            prompts = accelerate.utils.gather_object(inputs["prompt"])

            if state.is_local_process_zero:
                nll_sum += nll_loss.sum().item()
                total_samples += generation_sizes.size(0)
                sample_length_sum += generation_sizes.sum().item()
                generation_strs = tokenizer.batch_decode(generation_ids, skip_special_tokens=True)
                all_prompts.extend(prompts)
                all_generations.extend(generation_strs)

        if state.is_world_process_zero:
            # gather_for_metrics doesn't work for list of strings?
            gold_log = {
                "eval/perplexity": math.exp(nll_sum / total_samples),
                "eval/gold_sample_length": sample_length_sum / total_samples,
            }
            for key, value in gold_log.items():
                print(f"{key}: {value}")
            if state.epoch:
                gold_log["epoch"] = round(state.epoch, 2)
                gold_log["step"] = state.global_step

            if self.log_n_samples_during_eval:
                samples_to_log = [
                    [prompt, generation[len(prompt) :]]
                    for prompt, generation in zip(
                        all_prompts[: self.log_n_samples_during_eval],
                        all_generations[: self.log_n_samples_during_eval],
                    )
                ]
                gold_log["gold_log"] = (
                    wandb.Table(
                        columns=["Prompt", "Policy"],
                        rows=samples_to_log,
                    ),
                )

            wandb.log(gold_log)
            generation_ds = Dataset.from_dict({"generations": all_generations})
            generation_ds.push_to_hub(f"{self.hub_name}_generations", revision=str(state.global_step))

        self.completed_step = state.global_step

    def get_batch_samples(self, model, tokenizer, input_ids, attention_mask, return_ids=False) -> Tuple[str, str]:
        """Reduce inputs to unseen prompts, and maximum batch size if necessary
        Generate samples from the model and reference model for the given batch of inputs."""
        policy_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )

        # if self.ref_model is None:
        with self.accelerator.unwrap_model(model).disable_adapter():
            reference_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
            )
        # else:
        #     reference_output = self.ref_model.generate(
        #         **inputs,
        #         generation_config=self.generation_config,
        #     )

        policy_output = pad_to_length(policy_output, self.max_length, tokenizer.pad_token_id)
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, tokenizer.pad_token_id)
        reference_output_decoded = tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        if return_ids:
            return policy_output_decoded, reference_output_decoded, policy_output
        else:
            return policy_output_decoded, reference_output_decoded
