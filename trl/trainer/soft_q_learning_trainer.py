from typing import Callable, Dict, List, Optional, Tuple, Union
import copy
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from . import SoftQLearningConfig
from . import BaseTrainer
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper
import enum
from functools import partial
import torch.nn.functional as F
from typing import Union, Any
import numpy as np


BoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]

class ForwardMode(enum.Enum):
    SQL_OFF = "SQL_OFF"
    SQL_ON = "SQL_ON"

# forget config for now, transfer to config later
class SoftQLearningTrainer(BaseTrainer):
    r"""

    """
    def __init__(
            self,
            model: Union[PreTrainedModelWrapper, torch.nn.Module],
            target_model: Optional[Callable[[], torch.nn.Module]], # make this optional
            sql_loss_impl: str, # this could also be a callable
            target_update_method: Optional[str],
            target_learning_rate: float,
            mix_strategy: str,
            reward_shaping: bool,
            #reward_shaping_old_min: float,
            #reward_shaping_old_max: float,
            #reward_shaping_new_min: float,
            #reward_shaping_new_max: float,
            sql_loss_coefficients: Optional[float] = None,
            sql_loss_margin_constant: Optional[float] = None,
            sql_loss_margin_coefficient: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            beam_width: Optional[int] = None,
            reward_function: Callable[[List[str], List[str], List[str]], Tuple[FloatTensor, Dict[str, Any]]] = None,
            tokenizer = None,
            device = "cuda:0",
            reward_shaping_func: Callable[[FloatTensor], FloatTensor] = lambda r:r,
    ):
        self.model = model
        target_model_empty = target_model is None
        if target_model_empty:
            self.target_model = copy.deepcopy(self.model)
        else:
            self.target_model = target_model

        self.target_learning_rate = target_learning_rate
        self.target_sync_method = target_update_method
        self.mix_strategy = mix_strategy
        self.reward_function = reward_function
        self.tokenizer = tokenizer
        self.sql_loss_impl = sql_loss_impl
        self.reward_shaping = reward_shaping
        # self.reward_shaping_old_min = reward_shaping_old_min
        # self.reward_shaping_old_max = reward_shaping_old_max
        # self.reward_shaping_new_min = reward_shaping_new_min
        # self.reward_shaping_new_max = reward_shaping_new_max
        self.sql_loss_coefficients = sql_loss_coefficients
        self.sql_loss_margin_constant = sql_loss_margin_constant
        self.sql_loss_margin_coefficient = sql_loss_margin_coefficient
        self._top_k = top_k
        self._top_p = top_p
        self._beam_width = beam_width
        self.device = device
        self.reward_shaping_func = reward_shaping_func

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not target_model_empty:
            trainable_params.extend([p for p in self.target_model.parameters() if p.requires_grad])

        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    def step(self, batch, step):
        #if PREPROCESS_TARGET_TEXTS is True:
        #        if not isinstance(batch, tx.data.Batch):
        #            raise TypeError
        #        batch._batch["target_text"] = preprocess_target_texts(
        #            tokens_or_list_of_tokens=batch["target_text"],
        #            vocab=model._model.target_vocab,
        #            remove_special_tokens=False)

        # Do not sync when we learn the target model
        #if self.config.target_sync_method == "learn":
        #    if target_train_op is None:
        #        raise ValueError
        #    target_train_op()

        # If we use polyak-averaging
        # just do update every step
        # NOTE: wonder why this doesn't have a target_sync_steps portion
        if self.target_sync_method == "polyak":
            self.sync_target_model("polyak")

        elif (self.target_sync_method == "copy" and step % self.target_sync_steps == 0):
            self.sync_target_model("copy")

        candidate_modes = [ForwardMode.SQL_OFF, ForwardMode.SQL_ON]

        if self.mix_strategy == "alternate":
            modes = [candidate_modes[step % len(candidate_modes)]]

        if self.mix_strategy == "mix":
            modes = candidate_modes

        loss_list = []
        additional_info_list = []
        for mode in modes:
            _,_loss, _additional_info = self._forward_SQL(
                mode=mode,
                batch=batch)

            loss_list.append(_loss)
            additional_info_list.append(_additional_info)

        # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/2
        loss = torch.mean(torch.stack(loss_list))
        additional_info = unionize_dicts(additional_info_list)

        loss.backward()
        self.optimizer.step()

        batch_log = nested_detach_and_clone(additional_info, to_cpu=True)
        
        # return batch log here so IO portion can be taken care of outside this method
        return batch_log

    def sync_target_model(self, sync_type) -> None:
        # Do nothing
        # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py#L221
        if sync_type == "copy":
            self.target_model.load_state_dict(
                self.model.state_dict())

        # Target network update
        # Note that we are assuming `model.parameters()`
        # would yield the same parameter orders.
        # https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        elif sync_type == "polyak":
            for param_, param in zip(
                    self.target_model.parameters(),
                    self.model.parameters()):

                param_.data.copy_((1 - self.target_learning_rate) * param_ + self.target_learning_rate * param)
        else:
            # log warning
            # TODO: come back here
            return 

    def _forward_SQL(self, mode, batch):

        if mode == ForwardMode.SQL_OFF:
            # teacher forcing
            outputs = self.model(input_ids=batch[0].input_ids, decoder_input_ids=batch[1].input_ids)
            target_outputs = self.target_model(input_ids=batch[0].input_ids, decoder_input_ids=batch[1].input_ids)

            logits = outputs.logits.to(self.device)
            target_logits = target_outputs.logits.to(self.device)
            output_ids = batch[1].input_ids.to(self.device) # potential trouble spot
            sequence_lengths = batch[1].attention_mask.sum(dim=-1).to(self.device)

        elif mode == ForwardMode.SQL_ON:

            generation_length = batch[1].input_ids.shape[1]

            outputs = self.model.generate(
                batch[0].input_ids,
                do_sample=True,
                top_k=self._top_k,
                top_p=self._top_p,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=generation_length,
            )

            # generation of on-policy training data
            on_policy_training_data = (batch[0].input_ids, outputs.sequences[:,1:])

            # has to follow the steps taken by the model
            target_outputs = self.target_model(input_ids=on_policy_training_data[0], decoder_input_ids=on_policy_training_data[1]) 

            # possible problematic spot
            logits = torch.stack(outputs.scores, dim=1).to(self.device)
            target_logits = target_outputs.logits.to(self.device)
            output_ids = outputs.sequences[:,1:].contiguous().to(self.device)
            sequence_lengths = (output_ids != self.tokenizer.pad_token_id).sum(dim=-1).to(self.device)
             
        else:
            raise NotImplementedError
        
        predicted_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        input_texts = self.tokenizer.batch_decode(batch[0].input_ids, skip_special_tokens=True)
        target_texts = self.tokenizer.batch_decode(batch[1].input_ids, skip_special_tokens=True)

        raw_rewards, shaped_rewards, rewards_log = self.compute_rewards(
            input_texts,
            target_texts,
            predicted_texts)

        sql_loss, sql_loss_log = soft_q_loss_with_sparse_rewards(
            implementation=self.sql_loss_impl,
            logits = logits,
            logits_= target_logits,
            actions = output_ids,
            rewards = shaped_rewards,
            sequence_length=sequence_lengths,
            coefficient=self.sql_loss_coefficients,
            # Do not add margin losses unless the
            # actions are ground truth actions.
            margin_constant=(
                self.sql_loss_margin_constant
                if mode == ForwardMode.SQL_OFF else None),
            margin_coefficient=(
                self.sql_loss_margin_coefficient
                if mode == ForwardMode.SQL_OFF else None))

        add_prefix_to_dict_keys_inplace(
            rewards_log, prefix=f"{mode.value}/rewards/")
        add_prefix_to_dict_keys_inplace(
            sql_loss_log, prefix=f"{mode.value}/")

        sql_loss_log = unionize_dicts([
            rewards_log,
            sql_loss_log,
            {
                f"{mode.value}/rewards/raw": raw_rewards.mean(),
                f"{mode.value}/rewards/shaped": shaped_rewards.mean(),
            },
        ])

        return logits, sql_loss, sql_loss_log 

    def loss(self, *args):
        raise NotImplementedError("Not implemented")

    def compute_rewards(self, source_texts , target_texts,  output_texts):
        rewards_tensor, rewards_log = self.reward_function(
            sources=source_texts,
            targets=target_texts,
            predictions=output_texts,
            to_tensor=True,
            mode="train")

        rewards_tensor = rewards_tensor #.to(self.device)
        shaped_rewards_tensor = self.reward_shaping_func(rewards_tensor)
        return rewards_tensor, shaped_rewards_tensor, rewards_log

    def _save_pretrained(self, save_directory):
        raise NotImplementedError("Not implemented")

# ====================================================================================================

# = Soft Q Learning Loss Functions with Sparse Rewards (TODO: note about where it is copied from ) 

# =============================================


# TODO: find a pytorch implementation of this
def get_entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.

    Args:
        logits: Unscaled log probabilities.

    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, -1)
    return entropy

def unionize_dicts(dict_list):
    merged_dict = {}
    for dict_ in dict_list:
        merged_dict.update(dict_)
    return merged_dict

def gather_2d_on_last_dim(
        tensor: FloatTensor,
        index: LongTensor,
        shape: torch.Size
) -> FloatTensor:
    """Simplified version of `tf.gather_nd` in PyTorch"""
    flattened_tensor = tensor.view(-1, tensor.shape[-1])
    flattened_index = index.view(-1)
    flattened_gathered_tensor = flattened_tensor[
        torch.arange(flattened_index.shape[0]),
        flattened_index]
    return flattened_gathered_tensor.view(shape)

def add_prefix_to_dict_keys_inplace(
        d: Dict[str, Any],
        prefix: str,
        keys_to_exclude: Optional[List[str]] = None,
) -> None:

    # https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
    keys = list(d.keys())
    for key in keys:
        if keys_to_exclude is not None and key in keys_to_exclude:
            continue

        new_key = f"{prefix}{key}"
        d[new_key] = d.pop(key)


def soft_q_loss_with_sparse_rewards(
        implementation: str,
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[FloatTensor, Dict[str, Any]]:
    """Soft Q Learning Loss Functions with Sparse Rewards

    Arguments:
        implementation: string, which loss function to use
        logits:          [batch_size, sequence_length, vocab_size]
        logits_:         [batch_size, sequence_length, vocab_size]
        logits_pi:       [batch_size, sequence_length, vocab_size]
        actions:         [batch_size, sequence_length]
        rewards:         [batch_size]
        sequence_length: [batch_size]
    """
    if implementation not in ["v0", "v1", "v2", "v3", "v2_v2r", "v3_v3r", "v2_v2r_v3_v3r"]:
        raise ValueError

    if not torch.is_tensor(rewards):
        raise TypeError

    if rewards.ndim != 1 or logits.shape[0] != rewards.shape[0]:
        raise ValueError

    if implementation == "v0":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_0

    elif implementation == "v1":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_1

    elif implementation == "v2":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_2

    if implementation == "v3":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_3

    if implementation == "v2_v2r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_2_2_reversed,
            coefficient=coefficient,
            margin_constant=margin_constant,
            margin_coefficient=margin_coefficient)

    if implementation == "v3_v3r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_3_3_reversed,
            coefficient=coefficient)

    if implementation == "v2_v2r_v3_v3r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_2_2_reversed_3_3_reversed,
            coefficient=coefficient)

    if logits.shape != logits_.shape:
        raise ValueError(
            f"`logits.shape` = {logits.shape}, but "
            f"`logits_.shape` = {logits_.shape}")

    raw_losses, quantities_to_log = _sql_loss_func(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length
    )

    loss = mask_and_reduce(
        sequence=raw_losses,
        sequence_length=sequence_length)
    loss_log = {
        "loss": loss,
        "sequence_length": sequence_length.float().mean(),
        "loss-normalized": mask_and_reduce(
            sequence=raw_losses,
            sequence_length=sequence_length,
            average_across_timesteps=True,
            sum_over_timesteps=False),
    }

    #for key, value in quantities_to_log.items():
    #    masked_mean, masked_min, masked_max = get_masked_mean_min_max(
    #        value, lengths=sequence_length)
    #    loss_log[f"{key}/min"] = masked_min
    #    loss_log[f"{key}/max"] = masked_max
    #    loss_log[f"{key}/mean"] = masked_mean

    return loss, loss_log


def soft_q_loss_with_sparse_rewards_1(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    Q = gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    # use `V` from the target if available
    V_ = logits_.logsumexp(dim=-1)

    # Build the target `= V_t+1 + r`
    # where we assume the rewards to be sparse
    # i.e., only comes at the final step
    Q_ = torch.zeros_like(Q)
    Q_[:, :-1] = V_[:, 1:]
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards

    raw_losses = F.mse_loss(Q, Q_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": logits.logsumexp(dim=-1),
        "Q_": Q_,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        _recover_mle: bool = False,
) -> Tuple[FloatTensor, Dict[str, Any]]:
    
    Q = gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V = logits.logsumexp(dim=-1)
    A = Q - V

    # Target outputs
    Q_ = torch.zeros_like(Q)
    A_ = torch.zeros_like(Q)
    V_ = logits_.logsumexp(dim=-1)
    Q_[:, :-1] = V_[:, 1:]
    A_[:, :-1] = V_[:, 1:] - V_[:, :-1]
    # Terminal V-target is the last V-target before
    # the episode ends, thus depends on `sequence_length`
    terminal_V_ = V_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1]
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards
    A_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards - terminal_V_

    #if _recover_mle is True:
    #    sql_utils.colorful_warning("Recover-MLE Mode", bg="red")
    #    A_ = A.detach() + 1

    raw_losses = F.mse_loss(A, A_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "Q_": Q_,
        "V_": V_,
        "A_": A_,
        "H": get_entropy(logits),
        "H_":get_entropy(logits_),
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_3(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        freeze_future_steps: bool = False,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    Q = gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V = logits.logsumexp(dim=-1)
    A = Q - V

    # Target outputs
    V_ = logits_.logsumexp(dim=-1)

    A2 = masked_reverse_cumsum(
        A,
        lengths=sequence_length,
        dim=-1)

    if freeze_future_steps is True:
        # This line of code essentially
        # decompose `A` (with gradient)
        # and cumsum of future `A`
        # (without gradient)
        A2 = (A2 - A).detach() + A

    raw_losses = F.mse_loss(
        A2, rewards.view(-1, 1) - V_,
        reduction="none")

    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2_2_reversed(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    raw_losses_2, quantities_to_log_2 = soft_q_loss_with_sparse_rewards_2(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length)

    add_prefix_to_dict_keys_inplace(
        quantities_to_log_2, prefix="0/")

    if coefficient is not None:
        raw_losses_2_r, quantities_to_log_2_r = soft_q_loss_with_sparse_rewards_2(
            logits=logits_,
            logits_=logits,
            actions=actions,
            rewards=rewards,
            sequence_length=sequence_length)

        raw_losses = (
            coefficient * raw_losses_2 +
            (1 - coefficient) * raw_losses_2_r)

        add_prefix_to_dict_keys_inplace(
            quantities_to_log_2_r, prefix="1/")

        quantities_to_log = unionize_dicts([
            quantities_to_log_2,
            quantities_to_log_2_r,
        ])

    else:
        raw_losses = raw_losses_2
        quantities_to_log = quantities_to_log_2

    if margin_constant is not None and margin_coefficient is not None:
        raw_losses_margin, quantities_to_log_margin = large_margin_classification_loss(
            logits=logits,
            expert_actions=actions,
            margin_constant=margin_constant)

        raw_losses = raw_losses + margin_coefficient * raw_losses_margin
        add_prefix_to_dict_keys_inplace(
            quantities_to_log_margin, prefix="margin/")
        quantities_to_log = unionize_dicts([
            quantities_to_log,
            quantities_to_log_margin,
        ])

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_3_3_reversed(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    raw_losses_3, quantities_to_log_3 = soft_q_loss_with_sparse_rewards_3(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length)

    add_prefix_to_dict_keys_inplace(
        quantities_to_log_3, prefix="0/")

    if coefficient is not None:
        raw_losses_3_r, quantities_to_log_3_r = soft_q_loss_with_sparse_rewards_3(
            logits=logits_,
            logits_=logits,
            actions=actions,
            rewards=rewards,
            sequence_length=sequence_length)

        raw_losses = (
            coefficient * raw_losses_3 +
            (1 - coefficient) * raw_losses_3_r)

        add_prefix_to_dict_keys_inplace(
            quantities_to_log_3_r, prefix="1/")

        quantities_to_log = unionize_dicts([
            quantities_to_log_3,
            quantities_to_log_3_r,
        ])
    else:
        raw_losses = raw_losses_3
        quantities_to_log = quantities_to_log_3

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2_2_reversed_3_3_reversed(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    raw_losses_2, quantities_to_log_2 = soft_q_loss_with_sparse_rewards_2_2_reversed(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length,
        coefficient=coefficient)

    raw_losses_3, quantities_to_log_3 = soft_q_loss_with_sparse_rewards_3_3_reversed(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length,
        coefficient=coefficient)

    raw_losses = (raw_losses_2 + raw_losses_3) / 2

    add_prefix_to_dict_keys_inplace(
        quantities_to_log_2, prefix="v2/")
    add_prefix_to_dict_keys_inplace(
        quantities_to_log_3, prefix="v3/")
    quantities_to_log = unionize_dicts([
        quantities_to_log_2,
        quantities_to_log_3,
    ])
    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_0(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
) -> Tuple[FloatTensor, Dict[str, Any]]:
    V = logits.logsumexp(dim=-1)
    V_ = logits_.logsumexp(dim=-1)
    raw_losses = F.mse_loss(V, V_, reduction="none")
    quantities_to_log = {
        "V": V,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def large_margin_classification_loss(
        logits: FloatTensor,
        expert_actions: LongTensor,
        margin_constant: float,
) -> Tuple[FloatTensor, Dict[str, Any]]:
    """Deep Q-learning from Demonstrations

    Arguments:
        logits: [batch_size, sequence_length, vocab_size]
        expert_actions: [batch_size, sequence_length]
    """
    # [0, 0, 0, ..., 1, 1, 1, ..., N, N, N, ...]
    batch_indices = (
        torch
        .arange(expert_actions.shape[0])
        .repeat_interleave(expert_actions.shape[1], dim=0))

    # [0, 1, 2, ..., 0, 1, 2, ..., 0, 1, 2, ...]
    sequence_indices = (
        torch
        .arange(expert_actions.shape[1])
        .repeat(expert_actions.shape[0]))

    # indices for the expert actions
    indices = (
        batch_indices,
        sequence_indices,
        expert_actions.flatten())

    # get the margin, and mask margins of expert actions
    margin = margin_constant * torch.ones_like(logits)
    margin[indices] = 0

    # [batch_size, sequence_length]
    raw_losses = (
        (logits + margin).max(dim=-1).values -
        logits[indices].view(expert_actions.shape)
    )

    quantities_to_log = {
        "loss": raw_losses,
    }

    return raw_losses, quantities_to_log


def masked_reverse_cumsum(
        X: FloatTensor,
        lengths: LongTensor,
        dim: int
) -> FloatTensor:
    masked_X = X * sequence_mask(
        lengths,
        max_len=X.shape[1])

    return (masked_X
            .flip(dims=[dim])
            .cumsum(dim=dim)
            .flip(dims=[dim]))


def get_masked_mean_min_max(
        X: FloatTensor,
        lengths: LongTensor
) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
    if X.ndim != 2 and lengths.ndim != 1:
        raise ValueError

    if X.shape[0] != lengths.shape[0]:
        raise ValueError

    mask = get_lengths_mask(
        X=X,
        lengths=lengths)

    masked_min = X.masked_fill(~mask, np.inf).min(dim=1)
    masked_max = X.masked_fill(~mask, -np.inf).max(dim=1)
    masked_mean = mask_and_reduce(
        sequence=X,
        sequence_length=lengths,
        average_across_timesteps=True,
        sum_over_timesteps=False)

    return (masked_mean,
            masked_min.values.mean(),
            masked_max.values.mean())

def mask_and_reduce(sequence: torch.Tensor,
                    sequence_length: torch.LongTensor,
                    rank: int = 2,
                    average_across_batch: bool = True,
                    average_across_timesteps: bool = False,
                    sum_over_batch: bool = False,
                    sum_over_timesteps: bool = True,
                    dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths, and reduces (average or sum) away dimensions.

    This is a combination of :func:`~texar.torch.utils.shapes.mask_sequences`
    and :func:`~texar.torch.losses.losses_utils.reduce_batch_time`.

    Args:
        sequence: A tensor of sequence values.
            If `time_major=False` (default), this must be a tensor of shape
            `[batch_size, max_time, d_2, ..., d_rank]`, where the rank of
            the tensor is specified with :attr:`rank`.
            The batch and time dimensions are exchanged if `time_major` is True.
        sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero. If `None`,
            no masking is performed.
        rank (int): The rank of :attr:`sequence`. Must be >= 2. Default is 2,
            i.e., `sequence` is a 2D Tensor consisting of batch and time
            dimensions.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_remaining (bool): If set, average the sequence across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time.
        sum_over_timesteps (bool): If set, sum the sequence across the time
            dimension. Must not set `average_across_timesteps` and
            `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the sequence across the batch
            dimension. Must not set `average_across_batch` and `sum_over_batch`
            at the same time.
        sum_over_remaining (bool): If set, sum the sequence across the remaining
            dimension. Must not set `average_across_remaining` and
            `sum_over_remaining` at the same time.
        dtype (torch.dtype): The dtype of the returned mask.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape `[max_time, batch_size, ...]`.
            If `False` (default), `sequence` must have
            shape `[batch_size, max_time, ...]`.

    Returns:
        A tensor containing the masked and reduced sequence.
    """
    if rank < 2:
        raise ValueError('`rank` must be >= 2.')

    if sequence_length is not None:
        sequence = mask_sequences(sequence,
                                  sequence_length,
                                  dtype=dtype)

    sequence = reduce_batch_time(sequence,
                                 sequence_length,
                                 average_across_batch,
                                 average_across_timesteps,
                                 sum_over_batch,
                                 sum_over_timesteps)

    reduce_time = average_across_timesteps or sum_over_timesteps
    reduce_batch = average_across_batch or sum_over_batch

    return sequence

def reduce_batch_time(sequence: torch.Tensor,
                      sequence_length: torch.LongTensor,
                      average_across_batch: bool = True,
                      average_across_timesteps: bool = False,
                      sum_over_batch: bool = False,
                      sum_over_timesteps: bool = True) -> torch.Tensor:
    r"""Average or sum over the respective dimensions of :attr:`sequence`, which
    is of shape `[batch_size, max_time, ...]`.

    Assumes :attr:`sequence` has been properly masked according to
    :attr:`sequence_length`.

    Args:
        sequence: A tensor to reduce.
        sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero. If `None`,
            no masking is performed.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the sequence across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the sequence across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.

    Returns:
        A tensor with dimension reduction.
    """
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError("Only one of `average_across_timesteps` and "
                         "`sum_over_timesteps` can be set.")
    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and "
                         "`sum_over_batch` can be set.")

    if sum_over_timesteps:
        sequence = torch.sum(sequence, dim=1)
    elif average_across_timesteps:
        sequence = (torch.sum(sequence, dim=1).float() /
                    sequence_length.float())

    if sum_over_batch:
        sequence = torch.sum(sequence, dim=0)
    elif average_across_batch:
        sequence = torch.mean(sequence, dim=0)

    return sequence


def get_lengths_mask(
        X: FloatTensor,
        lengths: LongTensor
) -> FloatTensor:

    if any([
        X.ndim != 2,
        lengths.ndim != 1,
        X.shape[0] != lengths.shape[0]],
    ):
        raise ValueError

    return sequence_mask(lengths, max_len=X.shape[1])


def sequence_mask(lengths: Union[torch.LongTensor, List[int]],
                  max_len: Optional[int] = None,
                  dtype: Optional[torch.dtype] = None,
                  device: Optional[torch.device] = None) -> torch.ByteTensor:
    r"""Return a mask tensor representing the first N positions of each cell.

    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with

    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```

    Examples:

    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]

    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```

    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask

def mask_sequences(sequence: Union[torch.Tensor, List[int]],
                   sequence_length: Union[torch.LongTensor, List[int]],
                   dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    :attr:`sequence` and :attr:`sequence_length` can either be python
    arrays or Tensors, respectively. If both are Python arrays (or None), the
    return will be a Python array as well.

    Args:
        sequence: A Tensor or Python array of sequence values.
            If ``time_major==False`` (default), this must be a Tensor of shape
            ``[batch_size, max_time, ...]``. The batch and time dimension is
            exchanged if ``time_major==True``.
        sequence_length: A Tensor or python array of shape ``[batch_size]``.
            Time steps beyond the respective sequence lengths will be
            made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            ``[max_time, batch_size, ...]``.
            If `False` (default), :attr:`sequence` must have
            shape ``[batch_size, max_time, ...]``.

    Returns:
        The masked sequence, i.e., a Tensor or python array of the same shape
        as :attr:`sequence` but with masked-out entries (set to zero).

        If both :attr:`sequence` and :attr:`sequence_length` are python
        arrays, the returned value is a python array as well.
    """
    if not torch.is_tensor(sequence):
        sequence = torch.tensor(sequence, dtype=dtype)
    sequence: torch.Tensor

    rank = sequence.dim()
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")

    max_time = sequence.size(1)
    if dtype is None:
        dtype = sequence.dtype
    mask = sequence_mask(sequence_length, max_time, dtype=dtype)
    mask = mask.view(*mask.size(), *([1] * (rank - 2)))
    sequence = sequence * mask

    return sequence


def nested_detach_and_clone(obj: Any, to_cpu: bool = False, to_numpy: bool = False):
    if to_cpu is False and to_numpy is True:
        raise ValueError("Numpy has to be on CPU")

    def _operation(X: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
        _X = X.detach().clone()
        if to_cpu is True:
            _X = _X.cpu()

        if to_numpy is True:
            _X = _X.numpy()

        return _X

    return nested_tensor_operation(obj=obj, tensor_operation=_operation)


def nested_to_cuda(obj: Any):

    def _operation(X: torch.Tensor) -> torch.Tensor:
        return X.cuda()

    return nested_tensor_operation(obj=obj, tensor_operation=_operation)


def nested_tensor_operation(obj: Any, tensor_operation: Callable[[torch.Tensor], Any]) -> Any:
    """Nested Application of `detach().clone()`.

       This function will remove gradients and reference.
    """
    if isinstance(obj, (list, tuple)):
        return [
            nested_tensor_operation(
                obj=_obj,
                tensor_operation=tensor_operation)
            for _obj in obj]

    if isinstance(obj, dict):
        new_dict_obj = {}
        for key, val in obj.items():
            if not isinstance(key, str):
                raise NotImplementedError

            new_dict_obj[key] = nested_tensor_operation(
                obj=val,
                tensor_operation=tensor_operation)

        return new_dict_obj

    if isinstance(obj, torch.Tensor):
        return tensor_operation(obj)

    if obj is None:
        return obj

    if isinstance(obj, bool):
        # Special handling, since `bool` is subclass of `int
        # https://stackoverflow.com/questions/37888620/comparing-boolean-and-int-using-isinstance
        return obj

    if isinstance(obj, (int, float, str)):
        return obj

    raise TypeError(f"Unrecognized type {type(obj)}")