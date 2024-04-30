import time
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedModel

from trl.trainer import WinRateCallback, MockJudge, PairRMJudge
from accelerate.utils import is_deepspeed_available

from trl import DPOTrainer

if is_deepspeed_available():
    import deepspeed

def remove_hooks(model):
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def add_hooks(model):
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
def prepare_model_for_generation(model, accelerator):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.gradient_checkpointing_disable()
    unwrapped_model.config.use_cache = True

    if (
        hasattr(accelerator.state, "deepspeed_plugin")
        and accelerator.state.deepspeed_plugin is not None
        and accelerator.state.deepspeed_plugin.zero_stage == 3
    ):
        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield model
            add_hooks(model)
    else:
        yield unwrapped_model
        unwrapped_model.gradient_checkpointing_enable()
        unwrapped_model.config.use_cache = False

class OnlineDPOTrainer(DPOTrainer):
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        inputs = self.generate_annotate(model, inputs)
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def generate_annotate(self, model, inputs):
        # TODO: Make this part of the ODPOConfig class?
        generation_config = GenerationConfig(
            temperature=0.9,
            do_sample=True,
            num_return_sequences=2,  # Generate 2 reponses for each prompt
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # remove_padding=False,
            max_new_tokens=512,
        )
        with prepare_model_for_generation(model, self.accelerator) as unwrapped_model:
            start_time = time.time()
            generations = unwrapped_model.generate(
                inputs["prompt_input_ids"],
                attention_mask=inputs["prompt_attention_mask"],
                generation_config=generation_config,
            )
            generation_time = torch.tensor([time.time() - start_time]).to(self.accelerator.device)

        generation_time_gather = self.accelerator.gather(generation_time)
        if self.accelerator.is_main_process:
            print(
                f"Generation time: {generation_time_gather.mean().item():.2f} seconds for {len(generations)} generations"
            )

        padded_prompt_length = inputs["prompt_input_ids"].shape[1]
        # remove the prompt from the generated responses
        generations = generations[:, padded_prompt_length:]
        # decode the generated responses
        text_generations = self.tokenizer.batch_decode(generations, skip_special_tokens=True)

        # combine the original prompt
        annotation_batch = {"prompts": [], "completions": []}
        # eos_token = self.tokenizer.eos_token
        for i in range(self._train_batch_size):
            prompt = inputs["prompt"][i]
            response0 = text_generations[i * 2]  # append eos token for annotator?
            response1 = text_generations[i * 2 + 1]
            annotation_batch["prompts"].append(prompt)
            annotation_batch["completions"].append([response0, response1])
        results = self.annotator.judge_batch(annotation_batch["prompts"], annotation_batch["completions"])
        # TODO: Make this is a bit cleaner
        # if type(self.annotator) is FutureAnnotator:
        #     # annotate the responses with the GPT3.5 annotator
        #     results = self.annotator.judge_batch(annotation_batch["prompts"], annotation_batch["completions"])

        # elif type(self.annotator) is PairRMAnnotator:
        # convs_a = []
        # convs_b = []
        # for i in range(self._train_batch_size):
        #     response0 = inputs["messages"][i][:-1]
        #     response1 = inputs["messages"][i][:-1]
        #     response0.append({"role": "assistant", "content": text_generations[i * 2]})
        #     response1.append({"role": "assistant", "content": text_generations[i * 2 + 1]})
        #     if response0[0]["role"] == "system":  #  PAIRRM is not compatible with system prompts
        #         response0 = response0[1:]
        #         response1 = response1[1:]

        #     convs_a.append(response0)
        #     convs_b.append(response1)
        # results = self.annotator.judge_batch(convs_a, convs_b)
        # else:
        #     raise ValueError("Unknown Annotator", self.annotator)

        optimization_batch = []
        for chosen_index, prompt, completion in zip(
            results, annotation_batch["prompts"], annotation_batch["completions"]
        ):
            chosen = completion[chosen_index]
            rejected = completion[1 - chosen_index]

            sample = {
                "prompt": prompt.removeprefix(
                    self.tokenizer.bos_token
                ),  # the BOS token was added again in the tokenize_row method
                "chosen": chosen,
                "rejected": rejected,
            }
            # call the parent classes tokenize row that works with prompt, chosen, rejected
            sample_tokenized = super().tokenize_row(sample, self.model)
            optimization_batch.append(sample_tokenized)

        padded_batch = self.data_collator(optimization_batch)
        padded_batch = self._prepare_inputs(padded_batch)  # move to GPU
        return padded_batch

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]

        if self.is_encoder_decoder:
            raise NotImplementedError("Encoder-decoder models are not supported yet")

        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        # add BOS token to head of prompt
        prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]

        # if sequence is too long, truncate the prompt

        if self.truncation_mode == "keep_start":
            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                prompt_tokens[k] = prompt_tokens[k][: self.max_prompt_length]
        elif self.truncation_mode == "keep_end":
            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                prompt_tokens[k] = prompt_tokens[k][-self.max_prompt_length :]
        else:
            raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        for k, toks in {"": prompt_tokens}.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        return batch

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # For some reason, the model provided to this method is not wrapped in deepspeed, so we do it manually here.
        if self.model_wrapped is not None:
            model = self.model_wrapped

        inputs = self.generate_annotate(model, inputs)
        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)
