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

import transformers
from packaging.version import Version
from transformers import AutoConfig, PretrainedConfig


class JevaModelConfig(PretrainedConfig):
    """
    Configuration of a Jeva encoder and matrix decision stack.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            Configuration of the shared text encoder. Defaults to a Qwen2 configuration.
        projection_dim (`int`, *optional*, defaults to `768`):
            Width of the state and option projections and decision stack.
        decision_depth (`int`, *optional*, defaults to `1`):
            Number of state-update blocks.
        decision_num_heads (`int`, *optional*, defaults to `8`):
            Number of heads in each decision block.
        decision_ffn_ratio (`float`, *optional*, defaults to `4.0`):
            Feed-forward expansion ratio. Set to zero to omit the feed-forward layers.
        decision_dropout (`float`, *optional*, defaults to `0.0`):
            Dropout in the decision feed-forward layers.
        decision_zero_init_ffn (`bool`, *optional*, defaults to `True`):
            Initialize feed-forward output projections to zero.
        decision_final_norm (`bool`, *optional*, defaults to `False`):
            Apply RMS normalization before reading the decision scores.
        normalize_state (`bool`, *optional*, defaults to `False`):
            Normalize the initial state representation.
        pooling (`str`, *optional*, defaults to `"last"`):
            Encoder pooling: `"last"`, `"mean"`, or `"cls"`.
        tau_set_init (`float`, *optional*, defaults to `0.15`):
            Initial state-update attention temperature.
        tau_dec_init (`float`, *optional*, defaults to `0.1`):
            Initial candidate-distribution temperature.
        tau_cov_init (`float`, *optional*, defaults to `0.1`):
            Initial coverage temperature.
        alpha_over_tau_init (`float`, *optional*, defaults to `0.6`):
            Initial ratio controlling the state-update step size.
        coverage_bias_init (`float`, *optional*, defaults to `0.0`):
            Initial coverage threshold.
        min_temperature (`float`, *optional*, defaults to `1e-4`):
            Lower bound of the learned temperatures.
        backbone_name_or_path (`str`, *optional*):
            Encoder identifier used to find its tokenizer before the first save.
        adapter_config (`dict`, *optional*):
            PEFT configuration for an adapter inside the shared encoder.
    """

    model_type = "jeva"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        projection_dim=768,
        decision_depth=1,
        decision_num_heads=8,
        decision_ffn_ratio=4.0,
        decision_dropout=0.0,
        decision_zero_init_ffn=True,
        decision_final_norm=False,
        normalize_state=False,
        pooling="last",
        tau_set_init=0.15,
        tau_dec_init=0.1,
        tau_cov_init=0.1,
        alpha_over_tau_init=0.6,
        coverage_bias_init=0.0,
        min_temperature=1e-4,
        backbone_name_or_path=None,
        adapter_config=None,
        **kwargs,
    ):
        if backbone_config is None:
            backbone_config = AutoConfig.for_model("qwen2")
        elif isinstance(backbone_config, dict):
            backbone_config = dict(backbone_config)
            backbone_config = AutoConfig.for_model(backbone_config.pop("model_type"), **backbone_config)
        self.backbone_config = backbone_config
        kwargs.setdefault("attn_implementation", backbone_config._attn_implementation)
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.decision_depth = decision_depth
        self.decision_num_heads = decision_num_heads
        self.decision_ffn_ratio = decision_ffn_ratio
        self.decision_dropout = decision_dropout
        self.decision_zero_init_ffn = decision_zero_init_ffn
        self.decision_final_norm = decision_final_norm
        self.normalize_state = normalize_state
        self.pooling = pooling
        self.tau_set_init = tau_set_init
        self.tau_dec_init = tau_dec_init
        self.tau_cov_init = tau_cov_init
        self.alpha_over_tau_init = alpha_over_tau_init
        self.coverage_bias_init = coverage_bias_init
        self.min_temperature = min_temperature
        self.backbone_name_or_path = backbone_name_or_path
        self.adapter_config = adapter_config
        if decision_depth < 1 or decision_num_heads < 1 or projection_dim % decision_num_heads:
            raise ValueError("decision_depth must be positive and decision_num_heads must divide projection_dim")
        if pooling not in {"last", "mean", "cls"}:
            raise ValueError("pooling must be last, mean, or cls")

        # Let Transformers apply the encoder's native sharding plans to the composite model.
        self.base_model_tp_plan = {
            f"backbone.{key}": value for key, value in (backbone_config.base_model_tp_plan or {}).items()
        }
        if Version(transformers.__version__) >= Version("5.21.0.dev0"):
            self.base_model_ep_plan = {
                f"backbone.{key}": value for key, value in (backbone_config.base_model_ep_plan or {}).items()
            }

    def get_text_config(self, decoder=False, encoder=False):
        return self.backbone_config

    def to_dict(self):
        output = super().to_dict()
        if self.adapter_config is not None:
            output["adapter_config"] = {
                key: list(value) if isinstance(value, set) else value for key, value in self.adapter_config.items()
            }
        return output
