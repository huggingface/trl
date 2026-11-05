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

import copy
import socket
import sys
from pathlib import Path

import torch
from datasets import Dataset
from examples.async_distillation_mopd.launcher import _spawn, _stop_process, wait_for_vllm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_peft_available

from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer

from ..testing_utils import require_peft, require_torch_accelerator, require_vllm


if is_peft_available():
    from peft import LoraConfig


@require_vllm
@require_peft
@require_torch_accelerator
def test_async_mopd_with_two_cross_tokenizer_teachers(tmp_path):
    """Real HTTP scoring, spawned worker, packed training, GA and adapter sync.

    One GPU is enough for these tiny real checkpoints: adapter upload avoids a
    same-GPU NCCL rendezvous. The Qwen student is scored by both Llama and
    SmolLM tokenizers. No trainer, tokenizer, loss, or model internals are mocked.
    """
    model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    student_tokenizer = AutoTokenizer.from_pretrained(model_id)
    paths = {"student": tmp_path / "student"}
    model.save_pretrained(paths["student"])
    student_tokenizer.save_pretrained(paths["student"])
    for name, tokenizer_name in (
        ("llama", "trl-internal-testing/tiny-LlamaForCausalLM-3.2"),
        ("smol", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        teacher = copy.deepcopy(model)
        teacher.resize_token_embeddings(len(tokenizer))
        teacher.config.eos_token_id = tokenizer.eos_token_id
        teacher.config.pad_token_id = tokenizer.pad_token_id
        teacher.generation_config.eos_token_id = tokenizer.eos_token_id
        teacher.generation_config.pad_token_id = tokenizer.pad_token_id
        paths[name] = tmp_path / name
        teacher.save_pretrained(paths[name])
        tokenizer.save_pretrained(paths[name])
    processes, urls = [], {}
    trainer = None
    try:
        for name, path in paths.items():
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            log = tmp_path / f"{name}.log"
            command = [
                str(Path(sys.executable).with_name("vllm")),
                "serve",
                str(path),
                "--port",
                str(port),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "1024",
                "--gpu-memory-utilization",
                "0.15",
                "--enforce-eager",
                "--max-num-seqs",
                "4",
                "--max-num-batched-tokens",
                "1024",
            ]
            if name == "student":
                command.extend(["--enable-lora", "--max-loras", "3", "--max-lora-rank", "8"])
            else:
                command.extend(["--logprobs-mode", "processed_logprobs", "--max-logprobs", "-1"])
            process = _spawn(
                command,
                log,
                env={
                    "VLLM_SERVER_DEV_MODE": "1",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                },
            )
            processes.append(process)
            urls[name] = f"http://127.0.0.1:{port}"
            wait_for_vllm(urls[name], process, log, deadline_s=300)
        dataset = Dataset.from_list(
            [
                {"prompt": [{"role": "user", "content": question}], "teacher_id": expert}
                for expert, question in (
                    ("llama", "What is two plus two?"),
                    ("smol", "Write a greeting."),
                    ("llama", "What is three plus four?"),
                    ("smol", "Describe a cat."),
                )
            ]
        )
        config = AsyncDistillationConfig(
            output_dir=str(tmp_path / "output"),
            distillation_method="csd",
            teacher_server_urls={key: urls[key] for key in ("llama", "smol")},
            teacher_tokenizers={key: str(paths[key]) for key in ("llama", "smol")},
            vllm_server_base_url=urls["student"],
            max_steps=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_completion_length=16,
            beta=0.5,
            teacher_top_k=8,
            token_budget=0,
            max_staleness=1,
            max_concurrent_rollouts=1,
            queue_maxsize=8,
            rollout_worker_timeout=600,
            vllm_server_timeout=300,
            save_strategy="no",
            report_to="none",
        )
        trainer = AsyncDistillationTrainer(
            model=str(paths["student"]),
            args=config,
            train_dataset=dataset,
            peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
        )
        before = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
        trainer.train()
        assert trainer.state.global_step == 2
        assert any(not torch.equal(before[name], p) for name, p in trainer.model.named_parameters() if p.requires_grad)
        history = [entry for entry in trainer.state.log_history if "teacher_chunk_frac/llama" in entry]
        assert history
        for expert in ("llama", "smol"):
            assert any(entry[f"teacher_chunk_frac/{expert}"] > 0 for entry in history)
            assert all(torch.isfinite(torch.tensor(entry[f"teacher_jsd/{expert}"])) for entry in history)
        assert any(entry["csd/alignment_coverage"] > 0 for entry in history)
    finally:
        if trainer is not None:
            trainer.end()
        for process in reversed(processes):
            _stop_process(process)
