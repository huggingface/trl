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

import pytest
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils.memory import release_memory
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM
from transformers.utils import is_peft_available

from trl import DistillationConfig, DistillationTrainer
from trl.experimental.gkd.gkd_trainer import GKDTrainer
from trl.trainer.distillation_trainer import _chunked_divergence_loss

from .testing_utils import (
    TrlTestCase,
    require_liger_kernel,
    require_peft,
    require_torch_accelerator,
    require_vision,
    require_vllm,
)


if is_peft_available():
    from peft import LoraConfig, PrefixTuningConfig, get_peft_model


def _reference_chunked_divergence(
    student_hidden,
    teacher_hidden,
    student_w,
    teacher_w,
    completion_mask,
    beta,
    num_items_in_batch=None,
    s_bias=None,
    t_bias=None,
    s_scale=1.0,
    t_scale=1.0,
    s_softcap=None,
    t_softcap=None,
    temperature=1.0,
):
    """Naive full-vocab reference for `_chunked_divergence_loss`: project the whole batch at once (no chunking) and
    build the JSD straight from the definition, so it shares neither the chunking nor `F.kl_div`'s argument order."""
    # Op order mirrors the loss's chunk body: matmul, + bias, * scale, softcap, / temperature.
    student_logits = student_hidden.float() @ student_w.float().t()
    teacher_logits = teacher_hidden.float() @ teacher_w.float().t()
    if s_bias is not None:
        student_logits = student_logits + s_bias.float()
    if t_bias is not None:
        teacher_logits = teacher_logits + t_bias.float()
    if s_scale != 1.0:
        student_logits = student_logits * s_scale
    if s_softcap is not None:
        student_logits = s_softcap * torch.tanh(student_logits / s_softcap)
    if t_scale != 1.0:
        teacher_logits = teacher_logits * t_scale
    if t_softcap is not None:
        teacher_logits = t_softcap * torch.tanh(teacher_logits / t_softcap)
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_probs, teacher_probs = student_log_probs.exp(), teacher_log_probs.exp()

    if beta == 0.0:  # forward KL: KL(teacher || student)
        per_element = teacher_probs * (teacher_log_probs - student_log_probs)
    elif beta == 1.0:  # reverse KL: KL(student || teacher)
        per_element = student_probs * (student_log_probs - teacher_log_probs)
    else:  # generalized JSD against the mixture M = (1 - beta) * student + beta * teacher
        mixture_log_probs = ((1 - beta) * student_probs + beta * teacher_probs).log()
        per_element = beta * (teacher_probs * (teacher_log_probs - mixture_log_probs)) + (1 - beta) * (
            student_probs * (student_log_probs - mixture_log_probs)
        )

    per_token = per_element.sum(dim=-1) * completion_mask  # (B, K)
    denom = completion_mask.sum() if num_items_in_batch is None else num_items_in_batch
    return per_token.sum() / denom


class TestChunkedDivergenceLoss(TrlTestCase):
    """Unit tests for the memory-efficient chunked JSD loss (`_chunked_divergence_loss`)."""

    def _inputs(self, B=2, K=6, H=8, V=17, n_masked=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        student_hidden = torch.randn(B, K, H, generator=g)
        teacher_hidden = torch.randn(B, K, H, generator=g)
        student_w = torch.randn(V, H, generator=g)
        teacher_w = torch.randn(V, H, generator=g)
        completion_mask = torch.ones(B, K)
        # Mask a few scattered positions so the packed masked tail is exercised.
        completion_mask.reshape(-1)[torch.randperm(B * K, generator=g)[:n_masked]] = 0
        return student_hidden, teacher_hidden, student_w, teacher_w, completion_mask

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("chunk_size", [3, 4, 100])  # divides / doesn't divide / exceeds n_valid (= 9)
    def test_matches_naive_full_vocab(self, beta, chunk_size):
        sh, th, sw, tw, mask = self._inputs()
        loss, _, n_valid = _chunked_divergence_loss(sh, th, sw, tw, mask, beta, chunk_size)
        expected = _reference_chunked_divergence(sh, th, sw, tw, mask, beta)
        torch.testing.assert_close(loss, expected)
        assert n_valid.item() == int(mask.sum().item())

    def test_different_teacher_student_hidden_sizes(self):
        # Teacher and student may have different hidden widths; only the vocabulary must match.
        g = torch.Generator().manual_seed(2)
        B, K, V = 2, 5, 13
        sh, th = torch.randn(B, K, 8, generator=g), torch.randn(B, K, 12, generator=g)
        sw, tw = torch.randn(V, 8, generator=g), torch.randn(V, 12, generator=g)
        mask = torch.ones(B, K)
        mask[1, -1] = 0
        loss, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=0.5, chunk_size=4)
        expected = _reference_chunked_divergence(sh, th, sw, tw, mask, beta=0.5)
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_applies_logit_scale_and_softcapping(self, beta):
        # Per-model `logit_scale` (Cohere) / `final_logit_softcapping` (Gemma) must be applied before the softmax.
        sh, th, sw, tw, mask = self._inputs()
        loss, _, _ = _chunked_divergence_loss(
            sh,
            th,
            sw,
            tw,
            mask,
            beta,
            chunk_size=4,
            student_logit_scale=0.7,
            teacher_logit_scale=1.3,
            student_final_logit_softcapping=50.0,
            teacher_final_logit_softcapping=30.0,
        )
        expected = _reference_chunked_divergence(
            sh, th, sw, tw, mask, beta, s_scale=0.7, t_scale=1.3, s_softcap=50.0, t_softcap=30.0
        )
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_applies_temperature(self, beta):
        # Softmax temperature softens both distributions before the divergence.
        sh, th, sw, tw, mask = self._inputs()
        loss, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta, chunk_size=4, temperature=2.0)
        expected = _reference_chunked_divergence(sh, th, sw, tw, mask, beta, temperature=2.0)
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_applies_lm_head_bias(self, beta):
        # An `lm_head` bias must be added to each chunk's logits (after the projection, before the softmax).
        sh, th, sw, tw, mask = self._inputs()
        g = torch.Generator().manual_seed(3)
        s_bias = torch.randn(sw.size(0), generator=g)
        t_bias = torch.randn(tw.size(0), generator=g)
        loss, _, _ = _chunked_divergence_loss(
            sh, th, sw, tw, mask, beta, chunk_size=4, student_lm_head_bias=s_bias, teacher_lm_head_bias=t_bias
        )
        expected = _reference_chunked_divergence(sh, th, sw, tw, mask, beta, s_bias=s_bias, t_bias=t_bias)
        torch.testing.assert_close(loss, expected)

    def test_beta_1_is_reverse_kl(self):
        sh, th, sw, tw, mask = self._inputs()
        loss, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=1.0, chunk_size=4)
        # Hand-rolled reverse KL: sum_x p_s * (log p_s - log p_t) over valid positions, normalized by n_valid.
        student_log_probs = torch.log_softmax(sh @ sw.t(), dim=-1)
        teacher_log_probs = torch.log_softmax(th @ tw.t(), dim=-1)
        per_token = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1) * mask
        expected = per_token.sum() / mask.sum()
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_parity_with_gkd(self, beta):
        # Identity lm_head so hidden states are the logits, then compare against GKD's full-vocab JSD (sum reduction).
        B, K, V = 2, 5, 11
        g = torch.Generator().manual_seed(1)
        student_logits = torch.randn(B, K, V, generator=g)
        teacher_logits = torch.randn(B, K, V, generator=g)
        eye = torch.eye(V)
        mask = torch.ones(B, K)
        mask[0, -1] = 0
        labels = torch.where(mask.bool(), torch.ones_like(mask, dtype=torch.long), torch.full_like(mask, -100).long())
        loss, _, _ = _chunked_divergence_loss(
            student_logits, teacher_logits, eye, eye, mask, beta, chunk_size=4, num_items_in_batch=1
        )
        gkd = GKDTrainer.generalized_jsd_loss(
            student_logits, teacher_logits, labels=labels, beta=beta, reduction="sum"
        )
        torch.testing.assert_close(loss, gkd)

    def test_masked_positions_ignored(self):
        sh, th, sw, tw, mask = self._inputs()
        loss_a, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=0.5, chunk_size=4)
        # Perturbing the hidden states at masked positions only must not change the loss.
        masked = mask.reshape(-1) == 0
        sh2, th2 = sh.clone().reshape(-1, sh.size(-1)), th.clone().reshape(-1, th.size(-1))
        sh2[masked] += 5.0
        th2[masked] += 5.0
        loss_b, _, _ = _chunked_divergence_loss(sh2.view_as(sh), th2.view_as(th), sw, tw, mask, beta=0.5, chunk_size=4)
        torch.testing.assert_close(loss_a, loss_b)

    def test_grads_flow_and_zero_at_masked(self):
        sh, th, sw, tw, mask = self._inputs()
        sh = sh.clone().requires_grad_(True)
        loss, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=0.5, chunk_size=4)
        loss.backward()
        grad = sh.grad.reshape(-1, sh.size(-1))
        valid = mask.reshape(-1) != 0
        assert (grad[valid].abs().sum(dim=-1) > 0).all()  # valid positions receive gradient
        assert torch.equal(grad[~valid], torch.zeros_like(grad[~valid]))  # masked positions get none

    def test_backward_matches_reference(self):
        # The chunked (checkpointed) backward must match a naive full-vocab autograd backward, not merely be non-null.
        # Only the student carries gradient (the teacher is a fixed target), so compare the student hidden + lm_head.
        sh, th, sw, tw, mask = self._inputs()
        sh_c, sw_c = sh.clone().requires_grad_(True), sw.clone().requires_grad_(True)
        loss_c, _, _ = _chunked_divergence_loss(sh_c, th, sw_c, tw, mask, beta=0.5, chunk_size=4)
        loss_c.backward()

        sh_r, sw_r = sh.clone().requires_grad_(True), sw.clone().requires_grad_(True)
        loss_r = _reference_chunked_divergence(sh_r, th, sw_r, tw, mask, beta=0.5)
        loss_r.backward()

        torch.testing.assert_close(sh_c.grad, sh_r.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sw_c.grad, sw_r.grad, atol=1e-5, rtol=1e-5)

    def test_fully_masked_batch_keeps_graph(self):
        # With every position masked the loss is 0, but backward must still touch every trainable student param
        # (hidden states, lm_head weight and bias) — otherwise DDP/FSDP synchronization hangs at the all-reduce.
        sh, th, sw, tw, _ = self._inputs()
        sh, sw = sh.clone().requires_grad_(True), sw.clone().requires_grad_(True)
        s_bias = torch.zeros(sw.size(0), requires_grad=True)
        mask = torch.zeros(sh.size(0), sh.size(1))
        loss, _, n_valid = _chunked_divergence_loss(
            sh, th, sw, tw, mask, beta=0.5, chunk_size=4, student_lm_head_bias=s_bias
        )
        assert n_valid.item() == 0
        assert torch.isfinite(loss)
        loss.backward()  # must not raise: the graph stays connected through the student hidden states + lm_head
        assert sh.grad is not None and sw.grad is not None and s_bias.grad is not None


class TestDistillationTrainer(TrlTestCase):
    def test_init_minimal(self):
        # Instantiate with only model, teacher_model and train_dataset.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            train_dataset=dataset,
        )

    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # The student entropy metric is logged (item 61).
        assert any("entropy" in entry for entry in trainer.state.log_history)

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_train_dataset_format(self, config_name):
        # Both the standard (plain-text prompt) and conversational (chat-message prompt) prompt-only formats train end
        # to end: the student generates on-policy completions, the teacher scores them.
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # The point of this test is only that both prompt-only formats are accepted and train end to end: a full step
        # ran and every parameter stayed finite. See `test_train` for the params-changed assertion.
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert all(torch.isfinite(param).all() for param in trainer.model.parameters())

    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            DistillationTrainer(
                model=model_id,
                teacher_model=model_id,
                args=DistillationConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = DistillationTrainer(
            model=model_id,
            teacher_model=model_id,
            args=DistillationConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    def test_train_with_eval(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            eval_strategy="steps",
            eval_steps=2,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[0]["eval_loss"] is not None

    def test_train_with_iterable_dataset(self):
        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train", streaming=True)

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_steps=4,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        # Iterable datasets force `dispatch_batches=False` so the stream can be sharded per process.
        assert trainer.args.accelerator_config.dispatch_batches is False

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_iterable_dataset_requires_dispatch_batches_false(self):
        # `dispatch_batches=True` is incompatible with iterable datasets (see get_train_dataloader).
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train", streaming=True)

        training_args = DistillationConfig(
            output_dir=self.tmp_dir, accelerator_config={"dispatch_batches": True}, report_to="none"
        )
        with pytest.raises(ValueError, match="Iterable datasets require `dispatch_batches=False`"):
            DistillationTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
                args=training_args,
                train_dataset=dataset,
            )

    def test_iterable_dataset_forces_num_workers_zero(self):
        # Multiple workers would shard and interleave the stream, breaking the generation-batch ordering that
        # `_prepare_inputs` relies on, so `dataloader_num_workers` is forced to 0 for iterable datasets.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train", streaming=True)

        training_args = DistillationConfig(
            output_dir=self.tmp_dir, dataloader_num_workers=4, max_steps=1, report_to="none"
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.args.dataloader_num_workers == 0

    def test_iterable_eval_keeps_map_style_train_workers(self):
        # A map-style train set keeps its workers even when the eval set is iterable; the single-worker restriction is
        # scoped to the iterable eval loader and not persisted.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        eval_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="test", streaming=True)

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            dataloader_num_workers=4,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        assert trainer.args.dataloader_num_workers == 4  # not disabled by the iterable eval set
        assert trainer.get_train_dataloader().num_workers == 4  # map-style train keeps its workers
        assert trainer.get_eval_dataloader().num_workers == 0  # iterable eval loader uses a single worker
        assert trainer.args.dataloader_num_workers == 4  # override is scoped, not persisted

    @pytest.mark.parametrize("train_dataset_type", ["dataset", "iterable_dataset"])
    def test_init_with_train_dataset(self, train_dataset_type):
        streaming = "iterable" in train_dataset_type
        train_dataset = load_dataset(
            "trl-internal-testing/zen", "standard_prompt_only", split="train", streaming=streaming
        )

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = DistillationConfig(output_dir=self.tmp_dir, max_steps=4 if streaming else -1, report_to="none")

        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=train_dataset,
        )
        assert trainer.train_dataset is train_dataset

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_prompt_only", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DistillationConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
        ],
    )
    def test_evaluate_with_eval_dataset(self, eval_dataset_type):
        # `evaluate` accepts a dataset passed directly, not only an `eval_dataset` set at init. Iterable datasets passed
        # this way must still be configured for the iterable path, since they are absent at init.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        streaming = "iterable" in eval_dataset_type
        eval_split = load_dataset(
            "trl-internal-testing/zen", "standard_prompt_only", split="test", streaming=streaming
        )
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            eval_dataset = eval_split
        elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
            dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
            eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=train_dataset,
        )

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_train_eval_on_start(self):
        # Evaluation runs before the first training step; the loss must be computable at that point.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            eval_strategy="steps",
            eval_steps=2,
            eval_on_start=True,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )
        trainer.train()

    def test_train_beta_non_zero(self):
        # `beta` is the JSD interpolation coefficient (0 = forward KL, 1 = reverse KL); an intermediate value exercises
        # the generalized-JSD-against-the-mixture branch of the loss end to end.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            beta=0.5,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_with_pad_to_multiple_of(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            pad_to_multiple_of=8,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_with_additional_generation_kwargs(self):
        """Test that training works with additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            top_p=0.9,
            top_k=10,
            min_p=0.01,
            repetition_penalty=1.1,
        )

        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_with_generation_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            # Pass gen kwargs
            generation_kwargs={"do_sample": True, "top_k": 50, "num_beams": 2, "length_penalty": -0.1},
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_with_chat_template_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            chat_template_kwargs={"enable_thinking": False},
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_multiple_dataloader_workers(self):
        # Pytest/CI often starts background threads before tests run. With Python 3.12, using the default "fork" start
        # method in a multi-threaded process emits a DeprecationWarning and may deadlock.
        #
        # We force "spawn" here to make multiprocessing safe under pytest when DataLoader workers are enabled. This is
        # test-environment–specific and not required by the training logic itself.
        #
        # This means the test does not cover "fork". However, "spawn" is stricter (requires full picklability and clean
        # state) and avoids fork-after-threads issues that pytest cannot reliably test anyway. Fork-specific behavior,
        # if needed, should be tested in a clean process outside pytest.
        torch.multiprocessing.set_start_method("spawn", force=True)

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            dataloader_num_workers=2,  # use multiple dataloader workers
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_with_gradient_accumulation(self):
        # Gradient accumulation exercises the `num_items_in_batch` global-token normalizer (#4719) end to end: the loss
        # over the accumulated micro-batches must be reduced by the global valid-token count, not the per-microbatch
        # mean, for the update to be correct.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            gradient_accumulation_steps=2,  # accumulate over 2 micro-batches
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_vllm
    @require_torch_accelerator
    def test_train_vllm(self):
        """Test that training works with vLLM for on-policy generation (colocate mode)."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            logging_strategy="no",
            use_vllm=True,
        )

        try:
            trainer = DistillationTrainer(
                model="Qwen/Qwen3-0.6B",  # tiny models are too small for vLLM
                teacher_model="Qwen/Qwen3-0.6B",
                args=training_args,
                train_dataset=dataset,
            )

            # Self-distillation (teacher == student) gives ~zero divergence and thus ~zero gradient, which would make
            # the params-changed check below vacuous. Diverge the teacher so the loss — and the update — clears fp noise.
            torch.manual_seed(0)
            with torch.no_grad():
                for p in trainer.teacher_model.parameters():
                    p.add_(0.5 * torch.randn_like(p))

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        except Exception as e:
            # If vLLM fails to initialize due to hardware constraints or other issues, that's expected
            if any(
                keyword in str(e).lower()
                for keyword in [
                    "outofmemoryerror",
                    "cuda",
                    "memory",
                    "insufficient",
                    "no such device",
                    "free memory",
                    "gpu memory utilization",
                    "decrease gpu memory",
                ]
            ):
                pytest.skip(f"Skipping vLLM training test due to hardware constraints: {e}")
            elif "KeyError" in str(e) and "RANK" in str(e):
                pytest.skip(f"Skipping vLLM training test due to environment setup issues: {e}")
            elif "ValueError" in str(e) and "memory" in str(e).lower():
                pytest.skip(f"Skipping vLLM training test due to memory constraints: {e}")
            else:
                raise

        release_memory(trainer.model, trainer)

    @require_peft
    def test_train_peft_config(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model,
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_train_peft_model(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        model = get_peft_model(model, LoraConfig())
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model,
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # In practice, this test is the same as `test_train_peft_config`, since gradient checkpointing is enabled by
    # default in `DistillationTrainer`. We keep it as a regression guard: if the default ever changes, we still
    # explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_peft_with_gradient_checkpointing(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            gradient_checkpointing=True,  # enable gradient checkpointing
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model,
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_peft_non_lm_head_target_allowed(self):
        # The lm_head guard must only fire when the adapter actually wraps lm_head. An adapter targeting other modules
        # (here q_proj/v_proj) leaves lm_head as a plain Linear, so the loss reads the real (frozen) head weight and
        # there is nothing to silently drop. Guards against an over-broad regression of the guard.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="float32")
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        DistillationTrainer(  # must construct without raising
            model=model,
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=DistillationConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=dataset,
            peft_config=LoraConfig(target_modules=["q_proj", "v_proj"]),
        )

    @require_peft
    def test_peft_modules_to_save_lm_head_allowed(self):
        # `modules_to_save=["lm_head"]` makes the head a fully trained copy (a ModulesToSaveWrapper, not a tuner layer),
        # so `get_output_embeddings().weight` resolves to the trained weight and the loss trains it correctly. The guard
        # keys on the head being a tuner layer, so this must stay unblocked.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="float32")
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        DistillationTrainer(  # must construct without raising
            model=model,
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=DistillationConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=dataset,
            peft_config=LoraConfig(target_modules=["q_proj", "v_proj"], modules_to_save=["lm_head"]),
        )

    @require_peft
    def test_peft_lm_head_adapter_raises(self):
        # Both loss paths read `lm_head.weight` directly, so a PEFT adapter on the head is silently ignored. Reject it.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with pytest.raises(ValueError, match="lm_head"):
            DistillationTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
                args=DistillationConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                peft_config=LoraConfig(target_modules=["q_proj", "lm_head"]),
            )

    @require_peft
    def test_peft_prompt_learning_raises(self):
        # Prompt-learning injects virtual tokens via `PeftModel.forward()`, which the backbone-direct loss bypasses.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with pytest.raises(ValueError, match="Prompt-learning"):
            DistillationTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
                args=DistillationConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                peft_config=PrefixTuningConfig(num_virtual_tokens=4, task_type="CAUSAL_LM"),
            )

    def test_teacher_vocab_size_mismatch_raises(self):
        # The local-teacher loss compares full next-token distributions, so student and teacher must share a
        # vocabulary. A teacher with a different vocab_size is rejected (use GOLD for cross-tokenizer distillation).
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with pytest.raises(ValueError, match="vocab_size"):
            DistillationTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                teacher_model="trl-internal-testing/tiny-LlamaForCausalLM-3.2",
                args=DistillationConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

    def test_teacher_model_init_kwargs_with_instantiated_teacher_raises(self):
        # `teacher_model_init_kwargs` only applies when the teacher is a model id; passing it alongside an already
        # instantiated teacher is a mistake worth surfacing.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with pytest.raises(ValueError, match="teacher_model_init_kwargs"):
            DistillationTrainer(
                model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                teacher_model=AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM"),
                args=DistillationConfig(
                    output_dir=self.tmp_dir, report_to="none", teacher_model_init_kwargs={"dtype": "float32"}
                ),
                train_dataset=dataset,
            )

    def test_loss_normalizes_by_num_items_in_batch(self):
        # When `num_items_in_batch` is passed (as under gradient accumulation), the divergence loss must be reduced as
        # sum / num_items_in_batch rather than the local per-microbatch mean. See issue #4719. The chunked JSD path
        # must honor `num_items_in_batch`.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=DistillationConfig(output_dir=self.tmp_dir, beta=0.5, report_to="none"),
            train_dataset=dataset,
        )

        device = trainer.accelerator.device
        prompt_length, completion_length = 4, 3
        vocab_size = trainer.model.config.vocab_size
        completion_mask = torch.ones(2, completion_length, dtype=torch.long, device=device)
        batch = {
            "prompt_ids": torch.randint(0, vocab_size, (2, prompt_length), device=device),
            "prompt_mask": torch.ones(2, prompt_length, dtype=torch.long, device=device),
            "completion_ids": torch.randint(0, vocab_size, (2, completion_length), device=device),
            "completion_mask": completion_mask,
        }

        # Number of valid (non-masked) completion tokens in the local batch.
        num_valid = completion_mask.sum()

        trainer.model.eval()
        with torch.no_grad():
            loss_mean = trainer.compute_loss(trainer.model, batch)  # num_items_in_batch=None -> local mean
            loss_global = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)
            loss_double = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid * 2)

        # With num_items_in_batch equal to the local valid-token count, sum/N equals the local mean.
        torch.testing.assert_close(loss_global, loss_mean, rtol=1e-4, atol=1e-6)
        # Doubling the global count exactly halves the loss (sum / num_items is linear in 1/num_items).
        torch.testing.assert_close(loss_double, loss_mean / 2, rtol=1e-4, atol=1e-6)

    def test_compute_loss_matches_full_logit_reference(self):
        # The chunked JSD path must produce the same loss as the full-logit JSD it replaced (the handover). This also
        # checks the wiring: the student stays the student, the teacher the teacher, and only completion positions score.
        # A non-symmetric `beta` (0.25, not 0.5) is used deliberately so a student/teacher swap would change the loss.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=DistillationConfig(output_dir=self.tmp_dir, beta=0.25, report_to="none"),
            train_dataset=dataset,
        )

        device = trainer.accelerator.device
        vocab_size = trainer.model.config.vocab_size
        prompt_length, completion_length = 4, 3
        batch = {
            "prompt_ids": torch.randint(0, vocab_size, (2, prompt_length), device=device),
            "prompt_mask": torch.ones(2, prompt_length, dtype=torch.long, device=device),
            "completion_ids": torch.randint(0, vocab_size, (2, completion_length), device=device),
            "completion_mask": torch.ones(2, completion_length, dtype=torch.long, device=device),
        }
        num_valid = batch["completion_mask"].sum()

        trainer.model.eval()
        with torch.no_grad():
            loss = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)

            # Full-logit reference (the pre-chunking loss): run both models' full forward, take the completion logits,
            # and compute the generalized JSD over the completion positions, normalized by num_items.
            input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
            attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
            keep = slice(-completion_length - 1, -1)
            s = trainer.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, keep, :]
            t = trainer.teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, keep, :]
            slp = torch.log_softmax(s.float(), dim=-1)
            tlp = torch.log_softmax(t.float(), dim=-1)
            beta_t = torch.tensor(0.25)
            mixture = torch.logsumexp(torch.stack([slp + torch.log1p(-beta_t), tlp + torch.log(beta_t)]), dim=0)
            jsd = 0.25 * F.kl_div(mixture, tlp, reduction="none", log_target=True) + 0.75 * F.kl_div(
                mixture, slp, reduction="none", log_target=True
            )
            reference = jsd.sum() / num_valid

        torch.testing.assert_close(loss, reference, rtol=1e-4, atol=1e-6)

    @require_liger_kernel
    @require_torch_accelerator
    def test_liger_loss_agrees_with_chunked(self):
        # The Liger fused path and the default chunked path compute the same JSD objective (they share the hidden-state
        # extraction and differ only in the final loss call), so they must agree on a fixed batch. Toggling
        # `use_liger_kernel` on one trainer keeps the same (un-patched) model, so only the loss kernel differs.
        from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        trainer = DistillationTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=DistillationConfig(output_dir=self.tmp_dir, beta=0.5, report_to="none"),
            train_dataset=dataset,
        )

        # The tiny models are randomly initialized, so their next-token distributions are nearly uniform over the
        # ~150k vocab. At that scale the JSD sits at the float32 noise floor, where the chunked and fused paths' matmul
        # reduction orders diverge by ~2x — a conditioning artifact, not a real objective difference. Scale the LM heads
        # to peak the distributions so the two paths are compared on a well-conditioned batch.
        with torch.no_grad():
            trainer.model.get_output_embeddings().weight.mul_(50.0)
            trainer.teacher_model.get_output_embeddings().weight.mul_(50.0)

        device = trainer.accelerator.device
        vocab_size = trainer.model.config.vocab_size
        gen = torch.Generator().manual_seed(1)
        prompt_length, completion_length = 4, 3
        batch = {
            "prompt_ids": torch.randint(0, vocab_size, (2, prompt_length), generator=gen).to(device),
            "prompt_mask": torch.ones(2, prompt_length, dtype=torch.long, device=device),
            "completion_ids": torch.randint(0, vocab_size, (2, completion_length), generator=gen).to(device),
            "completion_mask": torch.ones(2, completion_length, dtype=torch.long, device=device),
        }
        num_valid = batch["completion_mask"].sum()

        trainer.model.eval()
        with torch.no_grad():
            chunked_loss = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)
            trainer.use_liger_kernel = True
            trainer.liger_loss = LigerFusedLinearJSDLoss(
                beta=trainer.beta,
                ignore_index=-100,
                temperature=trainer.temperature,
                compiled=False,
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
            )
            liger_loss = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)

        torch.testing.assert_close(liger_loss, chunked_loss, rtol=1e-3, atol=1e-4)

    @require_liger_kernel
    def test_liger_incompatible_with_logit_softcapping_raises(self):
        # The Liger fused JSD kernel can't apply Cohere `logit_scale` / Gemma `final_logit_softcapping`, so unlike the
        # chunked path it would optimize a different objective than the model's real forward. Reject rather than train
        # silently wrong.
        student = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
        student.config.final_logit_softcapping = 30.0
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with pytest.raises(ValueError, match="final_logit_softcapping"):
            DistillationTrainer(
                model=student,
                teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
                args=DistillationConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none"),
                train_dataset=dataset,
            )

    @require_liger_kernel
    def test_liger_allows_none_logit_scale(self):
        # `logit_scale = None` (e.g. MPT) means unscaled, like `1.0`; the Liger guard must not reject it.
        student = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
        student.config.logit_scale = None
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        DistillationTrainer(  # must not raise
            model=student,
            teacher_model="trl-internal-testing/small-Qwen3ForCausalLM",
            args=DistillationConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none"),
            train_dataset=dataset,
        )

    @require_liger_kernel
    def test_liger_rejects_zero_logit_scale(self):
        # `logit_scale = 0.0` is a real (degenerate) scale, not "unscaled" — it zeroes the logits. The Liger kernel
        # can't apply it, so like any other non-1.0 scale it must be rejected, not silently read as 1.0.
        student = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        student.config.logit_scale = 0.0
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with pytest.raises(ValueError, match="logit_scale"):
            DistillationTrainer(
                model=student,
                teacher_model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=DistillationConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none"),
                train_dataset=dataset,
            )


@require_vision
class TestDistillationTrainerVLM(TrlTestCase):
    @pytest.mark.parametrize(
        "model_id",
        [
            pytest.param(
                "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("4.57.0"),
                    reason="transformers<4.57 Gemma3 image processor can't batch variable-size images",
                ),
            ),
            pytest.param(
                "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Gemma4 models were introduced in transformers-5.5.0",
                ),
            ),
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.2.0"),
                    reason="Qwen3.5 models were introduced in transformers-5.2.0",
                ),
            ),
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.2.0"),
                    reason="Qwen3.5 models were introduced in transformers-5.2.0",
                ),
            ),
            # "trl-internal-testing/tiny-SmolVLMForConditionalGeneration", seems not to support bf16 properly
        ],
    )
    def test_train_vlm(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model_id,
            teacher_model=model_id,  # self-distillation: no tiny+small VLM fixture pair exists
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # Self-distillation gives a near-zero teacher signal, so we assert the multimodal path ran end to end and the
        # loss stayed finite, rather than params-changed (see `test_train_dataset_format`).
        train_loss = trainer.state.log_history[-1]["train_loss"]
        assert train_loss is not None
        assert torch.isfinite(torch.tensor(train_loss))

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",  # image_grid_thw path
            pytest.param(
                "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",  # image_position_ids path
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("4.57.0"),
                    reason="transformers<4.57 Gemma3 image processor can't batch variable-size images",
                ),
            ),
        ],
    )
    def test_train_vlm_gradient_accumulation(self, model_id):
        # With gradient accumulation the packed `pixel_values` are split across micro-batches via `num_images`; train a
        # couple of steps to exercise that path (a missing `num_images` would misalign images and text).
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            gradient_accumulation_steps=2,  # split the packed pixel_values across micro-batches
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model_id,
            teacher_model=model_id,  # self-distillation: no tiny+small VLM fixture pair exists
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        train_loss = trainer.state.log_history[-1]["train_loss"]
        assert train_loss is not None
        assert torch.isfinite(torch.tensor(train_loss))

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Gemma4 models were introduced in transformers-5.5.0",
                ),
            ),
        ],
    )
    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
    def test_train_vlm_and_vllm(self, model_id) -> None:
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
            vllm_mode="server",
        )
        trainer = DistillationTrainer(
            model=model_id,
            teacher_model=model_id,  # self-distillation: no tiny+small VLM fixture pair exists
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # Self-distillation gives a near-zero teacher signal, so we assert the multimodal path ran end to end and the
        # loss stayed finite, rather than params-changed (see `test_train_dataset_format`).
        train_loss = trainer.state.log_history[-1]["train_loss"]
        assert train_loss is not None
        assert torch.isfinite(torch.tensor(train_loss))
