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

import functools
import signal
import warnings
from abc import abstractmethod
from collections.abc import Callable

import psutil
import pytest
import torch
import torch.nn as nn
from transformers import is_bitsandbytes_available, is_comet_available, is_sklearn_available, is_wandb_available
from transformers.testing_utils import backend_device_count, torch_device
from transformers.utils import (
    is_kernels_available,
    is_peft_available,
    is_rich_available,
    is_torch_available,
    is_torch_bf16_gpu_available,
    is_torch_xla_available,
    is_vision_available,
)

from trl.chat_template_utils import _SUPPORTS_RESPONSE_TEMPLATE
from trl.import_utils import (
    is_harbor_available,
    is_jmespath_available,
    is_joblib_available,
    is_liger_kernel_available,
    is_math_verify_available,
    is_mergekit_available,
    is_openreward_available,
    is_vllm_available,
)


require_bitsandbytes = pytest.mark.skipif(not is_bitsandbytes_available(), reason="test requires bitsandbytes")
require_comet = pytest.mark.skipif(not is_comet_available(), reason="test requires comet_ml")
require_harbor = pytest.mark.skipif(not is_harbor_available(), reason="test requires harbor")
require_kernels = pytest.mark.skipif(not is_kernels_available(), reason="test requires kernels")
require_liger_kernel = pytest.mark.skipif(not is_liger_kernel_available(), reason="test requires liger-kernel")
require_math_latex = pytest.mark.skipif(not is_math_verify_available(), reason="test requires math_verify")
require_mergekit = pytest.mark.skipif(not is_mergekit_available(), reason="test requires mergekit")
require_openreward = pytest.mark.skipif(not is_openreward_available(), reason="test requires openreward")
require_peft = pytest.mark.skipif(not is_peft_available(), reason="test requires peft")
# Response parsing needs jmespath only on transformers < 5.13, which ships the legacy `response_schema` parser; the
# new-style `response_template` parser doesn't use it. See `_SUPPORTS_RESPONSE_TEMPLATE`.
require_response_parsing = pytest.mark.skipif(
    not _SUPPORTS_RESPONSE_TEMPLATE and not is_jmespath_available(),
    reason="test requires jmespath for response parsing on transformers below 5.13.0",
)
require_rich = pytest.mark.skipif(not is_rich_available(), reason="test requires rich")
require_sklearn = pytest.mark.skipif(
    not (is_sklearn_available() and is_joblib_available()), reason="test requires sklearn"
)
require_torch_accelerator = pytest.mark.skipif(
    torch_device is None or torch_device == "cpu", reason="test requires accelerator"
)
require_torch_multi_accelerator = pytest.mark.skipif(
    not is_torch_available() or backend_device_count(torch_device) <= 1, reason="test requires multiple accelerators"
)
require_vision = pytest.mark.skipif(not is_vision_available(), reason="test requires vision")
require_vllm = pytest.mark.skipif(not is_vllm_available(), reason="test requires vllm")
require_wandb = pytest.mark.skipif(not is_wandb_available(), reason="test requires wandb")
require_no_wandb = pytest.mark.skipif(is_wandb_available(), reason="test requires no wandb")
require_3_accelerators = pytest.mark.skipif(
    not (getattr(torch, torch_device, torch.cuda).device_count() >= 3),
    reason=f"test requires at least 3 {torch_device}s",
)
# `Trainer` wraps the model in `nn.DataParallel` whenever more than one accelerator is visible and no distributed
# launcher is used. TRL trainers don't support it: they rewire `forward` on a single module instance, which
# `DataParallel` breaks by replicating the module and scattering the inputs across devices.
xfail_data_parallel = pytest.mark.xfail(
    is_torch_available() and backend_device_count(torch_device) > 1,
    reason="TRL trainers do not support nn.DataParallel (https://github.com/huggingface/trl/issues/6836)",
)


def is_bitsandbytes_multi_backend_available() -> bool:
    if is_bitsandbytes_available():
        import bitsandbytes as bnb

        return "multi_backend" in getattr(bnb, "features", set())
    return False


# Function ported from transformers.testing_utils before transformers#41283
require_torch_gpu_if_bnb_not_multi_backend_enabled = pytest.mark.skipif(
    not is_bitsandbytes_multi_backend_available() and not torch_device == "cuda",
    reason="test requires bitsandbytes multi-backend enabled or 'cuda' torch device",
)


def is_ampere_or_newer(device_index=0):
    if not torch.cuda.is_available():
        return False

    # "Ampere" is an NVIDIA architecture; an AMD (ROCm) GPU is never Ampere. On ROCm,
    # torch.cuda.get_device_capability returns the gfx version, which would spuriously compare >= (8, 0).
    if torch.version.hip is not None:
        return False

    major, minor = torch.cuda.get_device_capability(device_index)
    # Ampere starts at compute capability 8.0 (e.g., A100 = 8.0, RTX 30xx = 8.6)
    return (major, minor) >= (8, 0)


def is_bf16_supported() -> bool:
    """Whether the current device has hardware bf16 support for `bf16=True` training.

    `is_torch_bf16_gpu_available` already covers CUDA (Ampere or newer), XPU, HPU, NPU, etc., and XLA devices are added
    via `is_torch_xla_available`. On a CPU or a pre-Ampere GPU (e.g. T4), `bf16=True` raises "Your setup doesn't
    support bf16/gpu", so tests should pass `bf16=is_bf16_supported()` instead of a hard-coded `True`.

    `transformers`' `TrainingArguments` validation also accepts `bf16=True` when `use_cpu=True` (CPU training is an
    explicit opt-in). This helper deliberately does not treat that as supported, since it reports hardware bf16
    capability; tests setting `use_cpu=True` should not rely on it.
    """
    return is_torch_bf16_gpu_available() or is_torch_xla_available()


class TrlTestCase:
    @pytest.fixture(autouse=True)
    def set_tmp_dir(self, tmp_path):
        self.tmp_dir = str(tmp_path)


def ignore_warnings(message: str = None, category: type[Warning] = Warning) -> Callable:
    """
    Decorator to ignore warnings with a specific message and/or category.

    Args:
        message (`str`, *optional*):
            Regex pattern for the warning message to ignore. If `None`, all messages are ignored.
        category (`type[Warning]`, *optional*, defaults to `Warning`):
            Warning class to ignore. Defaults to `Warning`, which ignores all warnings.
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=message, category=category)
                return test_func(*args, **kwargs)

        return wrapper

    return decorator


def kill_process(process):
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=5)
        except psutil.TimeoutExpired:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
        process.wait(timeout=5)
    except psutil.TimeoutExpired:
        process.kill()
    except psutil.NoSuchProcess:
        pass


def assert_verbose_allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, max_print=5, extra_info=""):
    """
    Assert that two tensors are element-wise equal within a tolerance, providing detailed information about mismatches.

    Parameters:
    tensor1 (torch.Tensor): First tensor to compare.
    tensor2 (torch.Tensor): Second tensor to compare.
    rtol (float): Relative tolerance.
    atol (float): Absolute tolerance.
    max_print (int): Maximum number of mismatched elements to print.
    extra_info (str): Extra information to show at the start of the error message.

    Raises:
    AssertionError: If the tensors are not all close within the given tolerance.
    """
    # Check if the shapes of the tensors match
    if tensor1.shape != tensor2.shape:
        raise AssertionError("Input tensors must have the same shape.")

    # Calculate the difference between the tensors
    diff = torch.abs(tensor1 - tensor2)

    # Determine the tolerance
    tolerance = atol + rtol * torch.abs(tensor2)

    # Find tolerance mismatched elements
    tol_mismatched = diff > tolerance

    # Find nan mismatched elements
    nan_mismatched = torch.logical_xor(torch.isnan(tensor1), torch.isnan(tensor2))

    # Find +inf mismatched elements
    posinf_mismatched = torch.logical_xor(torch.isposinf(tensor1), torch.isposinf(tensor2))
    # Find -inf mismatched elements
    neginf_mismatched = torch.logical_xor(torch.isneginf(tensor1), torch.isneginf(tensor2))

    # Find all mismatched elements
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched)

    # Count the number of mismatched elements
    num_mismatched = mismatched.sum().item()

    # Check if all elements are close
    all_close = num_mismatched == 0

    # Raise AssertionError with detailed information if there are mismatches
    if not all_close and num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        print_count = min(max_print, num_mismatched)
        for index in mismatched_indices[:print_count]:
            i = tuple(index.tolist())
            mismatch_details.append(f"Mismatch at index {i}: tensor1[{i}] = {tensor1[i]}, tensor2[{i}] = {tensor2[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched elements.")

        raise AssertionError(extra_info + "\n".join(mismatch_details))


class HFAlignmentLoss:
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        ignore_index: int = -100,
        use_ref_model: bool = False,
        unpaired: bool = False,
        compute_nll_loss: bool = True,
        **kwargs,
    ):
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.use_ref_model = use_ref_model
        self.unpaired = unpaired
        self.compute_nll_loss = compute_nll_loss

    @abstractmethod
    def alignment_loss(self):
        pass

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of ignore_index are
                ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return
                the sum of the log probabilities of the (non-masked) tokens.
            is_encoder_decoder: Whether the model is an encoder-decoder model.
        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the
            given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != self.ignore_index

        # dummy token; we'll ignore the losses on these tokens later
        labels = torch.where(labels == self.ignore_index, 0, labels)

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def get_ref_logps(
        self,
        ref_input: torch.FloatTensor,
        ref_weight: torch.FloatTensor,
        target: torch.LongTensor,
        ref_bias: torch.FloatTensor,
        average_log_prob: bool = True,
        preference_labels: torch.Tensor = None,
    ):
        """Compute the log probabilities of the given labels under the given reference model."""

        with torch.no_grad():
            ref_logits = ref_input @ ref_weight.t()
            if ref_bias is not None:
                ref_logits = ref_logits + ref_bias
            ref_all_logps = self.get_batch_logps(ref_logits, target, average_log_prob=average_log_prob)

            if self.unpaired and preference_labels is not None:
                # Split based on preference labels
                return (
                    ref_all_logps[preference_labels],
                    ref_all_logps[~preference_labels],
                )
            else:
                # Original paired behavior - split in half
                return (
                    ref_all_logps[: ref_input.shape[0] // 2],
                    ref_all_logps[ref_input.shape[0] // 2 :],
                )

    def concatenated_forward(
        self,
        _input: torch.FloatTensor,
        weight: torch.FloatTensor,
        target: torch.LongTensor,
        bias: torch.FloatTensor | None = None,
        average_log_prob: bool = True,
        preference_labels: torch.Tensor = None,
        nll_target: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        len_chosen = _input.shape[0] // 2

        outputs = _input @ weight.t()
        if bias is not None:
            outputs = outputs + bias
        all_logits = outputs.float()

        def cross_entropy_loss(logits, labels):
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = nll_target if nll_target is not None else target
        chosen_nll_loss = torch.tensor(0.0, device=all_logits.device)
        if self.compute_nll_loss:
            chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            target,
            average_log_prob=average_log_prob,
        )

        if self.unpaired and preference_labels is not None:
            # Split based on labels tensor
            chosen_logps = all_logps[preference_labels]
            rejected_logps = all_logps[~preference_labels]
            chosen_logits = all_logits[preference_labels]
            rejected_logits = all_logits[~preference_labels]
        else:
            # Original paired behavior - split in half
            len_chosen = _input.shape[0] // 2
            chosen_logps = all_logps[:len_chosen]
            rejected_logps = all_logps[len_chosen:]
            chosen_logits = all_logits[:len_chosen]
            rejected_logits = all_logits[len_chosen:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_nll_loss,
        )

    def get_batch_loss_metrics(
        self,
        weight: torch.FloatTensor,
        _input: torch.FloatTensor,
        target: torch.LongTensor,
        bias: torch.FloatTensor = None,
        ref_input: torch.FloatTensor = None,
        ref_weight: torch.FloatTensor = None,
        ref_bias: torch.FloatTensor = None,
        average_log_prob: bool = True,
        preference_labels: torch.Tensor = None,
        nll_target: torch.LongTensor = None,
        **loss_kwargs,
    ):
        """Compute the loss metrics for the given batch of inputs for train or test."""
        forward_output = self.concatenated_forward(
            _input, weight, target, bias, average_log_prob, preference_labels, nll_target
        )
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]

        if self.use_ref_model:
            ref_chosen_logps, ref_rejected_logps = self.get_ref_logps(
                ref_input,
                ref_weight,
                target,
                ref_bias,
                average_log_prob,
                preference_labels,
            )
            loss_kwargs["ref_chosen_logps"] = ref_chosen_logps
            loss_kwargs["ref_rejected_logps"] = ref_rejected_logps
        alignment_loss_outputs = self.alignment_loss(policy_chosen_logps, policy_rejected_logps, **loss_kwargs)
        if isinstance(alignment_loss_outputs, tuple):
            losses, *aggregated_aux_outputs = alignment_loss_outputs
        else:
            losses, aggregated_aux_outputs = alignment_loss_outputs, []

        loss = policy_nll_loss * self.alpha + losses.mean()

        if not self.unpaired:
            return_vars = (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits.detach().mean(),
                policy_rejected_logits.detach().mean(),
                policy_nll_loss,
            )
            return loss, (*return_vars, *aggregated_aux_outputs)
        else:
            return_vars = (
                policy_chosen_logps.detach().sum(),
                policy_rejected_logps.detach().sum(),
                policy_chosen_logits.detach().sum(),
                policy_rejected_logits.detach().sum(),
            )
            return loss, (*return_vars, *aggregated_aux_outputs)


class HFDistillationLoss:
    def __init__(
        self,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1,
    ):
        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.ignore_index = ignore_index
        self.temperature = temperature

    @abstractmethod
    def distillation_loss(self, student_logits, teacher_logits, **loss_kwargs):
        """Abstract method for computing distillation loss."""
        pass

    def concatenated_forward(
        self,
        student_input: torch.FloatTensor,
        student_weight: torch.FloatTensor,
        teacher_input: torch.FloatTensor,
        teacher_weight: torch.FloatTensor,
        target: torch.LongTensor,
        student_bias: torch.FloatTensor = None,
        teacher_bias: torch.FloatTensor = None,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Compute forward pass for both student and teacher models."""

        student_batch_seq_len_size, student_hidden_size = student_input.shape
        student_input_reshaped = student_input.view(-1, student_hidden_size)
        teacher_batch_seq_len_size, teacher_hidden_size = teacher_input.shape
        teacher_input_reshaped = teacher_input.view(-1, teacher_hidden_size)

        student_outputs = student_input_reshaped @ student_weight.t()
        if student_bias is not None:
            student_outputs = student_outputs + student_bias

        with torch.no_grad():
            teacher_outputs = teacher_input_reshaped @ teacher_weight.t()
            if teacher_bias is not None:
                teacher_outputs = teacher_outputs + teacher_bias

        student_logits = student_outputs.view(student_batch_seq_len_size, -1).float()
        teacher_logits = teacher_outputs.view(teacher_batch_seq_len_size, -1).float()

        if torch.all(target == self.ignore_index):
            return torch.tensor(0.0)

        def cross_entropy_loss(logits, labels):
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = target
        ce_loss = cross_entropy_loss(
            student_logits.view(-1, student_logits.shape[-1]),
            labels.view(-1),
        )

        return (
            student_logits,
            teacher_logits,
            ce_loss,
        )

    def get_batch_loss_metrics(
        self,
        student_input: torch.FloatTensor,
        student_weight: torch.FloatTensor,
        teacher_input: torch.FloatTensor,
        teacher_weight: torch.FloatTensor,
        target: torch.LongTensor,
        student_bias: torch.FloatTensor = None,
        teacher_bias: torch.FloatTensor = None,
        **loss_kwargs,
    ):
        """Compute the distillation loss metrics for the given batch."""
        forward_output = self.concatenated_forward(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            target,
            student_bias,
            teacher_bias,
        )
        (
            student_logits,
            teacher_logits,
            hard_loss,
        ) = forward_output

        student_logits /= self.temperature
        teacher_logits /= self.temperature

        soft_loss = self.distillation_loss(
            student_logits, teacher_logits, target=target, ignore_index=self.ignore_index, **loss_kwargs
        )
        # full loss
        loss = self.weight_hard_loss * hard_loss + self.weight_soft_loss * soft_loss
        return loss
