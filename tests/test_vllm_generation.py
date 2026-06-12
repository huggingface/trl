import sys
from importlib import util as importlib_util
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch
from torch import nn

profiling_stub = ModuleType("trl.extras.profiling")
profiling_stub.ProfilingContext = object
sys.modules.setdefault("trl.extras.profiling", profiling_stub)

import_utils_stub = ModuleType("trl.import_utils")


def _is_package_available(package_name: str, return_version: bool = False):
    is_available = importlib_util.find_spec(package_name) is not None
    if return_version:
        return is_available, "0.0.0"
    return is_available


import_utils_stub._is_package_available = _is_package_available
import_utils_stub.is_requests_available = lambda: True
import_utils_stub.is_vllm_ascend_available = lambda: False
import_utils_stub.is_vllm_available = lambda: False
sys.modules.setdefault("trl.import_utils", import_utils_stub)

trainer_utils_stub = ModuleType("trl.trainer.utils")


def ensure_master_addr_port(*args: Any, **kwargs: Any) -> None:
    return None


trainer_utils_stub.ensure_master_addr_port = ensure_master_addr_port
sys.modules.setdefault("trl.trainer.utils", trainer_utils_stub)

from trl.generation.vllm_generation import VLLMGeneration


class ParameterLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class TinyQwen35Causal(nn.Module):
    def __init__(self, architectures=None):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = ParameterLeaf()
        self.lm_head = ParameterLeaf()
        self.config = SimpleNamespace(architectures=architectures)


class TinyQwen35Conditional(nn.Module):
    def __init__(self, architectures=None):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = ParameterLeaf()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = ParameterLeaf()
        self.lm_head = ParameterLeaf()
        self.config = SimpleNamespace(architectures=architectures or ["Qwen3_5ForConditionalGeneration"])


def make_vllm_generation(model: nn.Module, mode: str) -> tuple[VLLMGeneration, MagicMock]:
    generation = VLLMGeneration.__new__(VLLMGeneration)
    generation.model = model
    generation.accelerator = SimpleNamespace(
        is_main_process=True,
        state=SimpleNamespace(deepspeed_plugin=None),
    )
    generation.is_fsdp_enabled = False
    generation.mode = mode
    generation.enable_sleep_mode = False
    generation.vllm_client = SimpleNamespace(
        update_named_param=MagicMock(),
        reset_prefix_cache=MagicMock(),
    )
    llm_model = MagicMock()
    generation.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(model_runner=SimpleNamespace(model=llm_model))
            )
        ),
        reset_prefix_cache=MagicMock(),
    )
    return generation, llm_model


def test_fix_param_name_maps_qwen35_causal_text_only_prefixes():
    generation, _ = make_vllm_generation(TinyQwen35Causal(architectures=["Qwen3_5ForCausalLM"]), mode="server")

    assert generation._fix_param_name_to_vllm("model.layers.weight") == "language_model.model.layers.weight"
    assert generation._fix_param_name_to_vllm("lm_head.weight") == "language_model.lm_head.weight"


def test_fix_param_name_maps_qwen35_conditional_prefixes():
    generation, _ = make_vllm_generation(TinyQwen35Conditional(), mode="server")

    assert generation._fix_param_name_to_vllm("model.visual.weight") == "visual.weight"
    assert (
        generation._fix_param_name_to_vllm("model.language_model.layers.weight")
        == "language_model.model.layers.weight"
    )
    assert generation._fix_param_name_to_vllm("lm_head.weight") == "language_model.lm_head.weight"


def test_fix_param_name_maps_qwen3vl_prefixes():
    model = TinyQwen35Conditional(architectures=["Qwen3VLForConditionalGeneration"])
    generation, _ = make_vllm_generation(model, mode="server")

    assert generation._fix_param_name_to_vllm("model.visual.weight") == "visual.weight"
    assert (
        generation._fix_param_name_to_vllm("model.language_model.layers.weight")
        == "language_model.model.layers.weight"
    )
    assert generation._fix_param_name_to_vllm("lm_head.weight") == "language_model.lm_head.weight"


def test_fix_param_name_keeps_unknown_architecture_unchanged():
    model = TinyQwen35Causal(architectures=["Qwen2ForCausalLM"])
    generation, _ = make_vllm_generation(model, mode="server")

    assert generation._fix_param_name_to_vllm("model.layers.weight") == "model.layers.weight"
    assert generation._fix_param_name_to_vllm("lm_head.weight") == "lm_head.weight"


def test_fix_param_name_strips_wrappers_before_remap():
    generation, _ = make_vllm_generation(TinyQwen35Causal(architectures=["Qwen3_5ForCausalLM"]), mode="server")

    assert (
        generation._fix_param_name_to_vllm(
            "_checkpoint_wrapped_module._fsdp_wrapped_module.model.layers.weight",
            extra_prefixes=["_fsdp_wrapped_module."],
        )
        == "language_model.model.layers.weight"
    )


def test_fix_param_name_uses_first_supported_architecture():
    model = TinyQwen35Causal(architectures=["UnknownArchitecture", "Qwen3_5ForCausalLM"])
    generation, _ = make_vllm_generation(model, mode="server")

    assert generation._fix_param_name_to_vllm("model.layers.weight") == "language_model.model.layers.weight"


def test_fix_param_name_keeps_missing_architectures_unchanged():
    generation, _ = make_vllm_generation(TinyQwen35Causal(architectures=None), mode="server")

    assert generation._fix_param_name_to_vllm("model.layers.weight") == "model.layers.weight"


def test_sync_weights_server_uses_qwen35_mapped_names():
    generation, _ = make_vllm_generation(TinyQwen35Causal(architectures=["Qwen3_5ForCausalLM"]), mode="server")

    generation.sync_weights()

    update_calls = generation.vllm_client.update_named_param.call_args_list
    synced_names = [call.args[0] for call in update_calls]
    assert synced_names == [
        "language_model.model.layers.weight",
        "language_model.lm_head.weight",
    ]


def test_sync_weights_colocate_uses_qwen35_mapped_names():
    generation, llm_model = make_vllm_generation(TinyQwen35Causal(architectures=["Qwen3_5ForCausalLM"]), mode="colocate")

    generation.sync_weights()

    load_calls = llm_model.load_weights.call_args_list
    synced_names = [call.args[0][0][0] for call in load_calls]
    assert synced_names == [
        "language_model.model.layers.weight",
        "language_model.lm_head.weight",
    ]


def test_sync_weights_keeps_non_qwen35_names_unchanged():
    model = TinyQwen35Causal(architectures=["Qwen2ForCausalLM"])
    generation, _ = make_vllm_generation(model, mode="server")

    generation.sync_weights()

    update_calls = generation.vllm_client.update_named_param.call_args_list
    synced_names = [call.args[0] for call in update_calls]
    assert synced_names == [
        "model.layers.weight",
        "lm_head.weight",
    ]
