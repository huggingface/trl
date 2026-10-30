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

# Jeva: System One Models from First Principles
# Copyright 2026 Qi Li and Xinchao Wang.

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import snapshot_download
from torch import nn
from torch.distributed.nn.functional import all_gather, all_reduce
from transformers import AutoModel, PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import ModelOutput

from .configuration_jeva import JevaModelConfig


def _log_softmax(logits):
    logits = logits.float()
    maximum = logits.argmax(dim=-1, keepdim=True)
    shifted = (logits - logits.gather(-1, maximum)).scatter(-1, maximum, 0.0)
    others = shifted.exp().scatter(-1, maximum, 0.0)
    return shifted - torch.log1p(others.sum(dim=-1, keepdim=True))


class JevaPositiveScalar(nn.Module):
    def __init__(self, value, minimum):
        super().__init__()
        self.minimum = minimum
        self.initial_value = math.log(math.expm1(max(value - minimum, 1e-6)))
        self.raw = nn.Parameter(torch.tensor([self.initial_value]))

    def forward(self):
        return self.minimum + F.softplus(self.raw)


class JevaSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.projection_dim
        width = round(dim * config.decision_ffn_ratio)
        self.gate_proj = nn.Linear(dim, width, bias=False)
        self.up_proj = nn.Linear(dim, width, bias=False)
        self.down_proj = nn.Linear(width, dim, bias=False)
        self.dropout = nn.Dropout(config.decision_dropout)
        self.zero_init_output = config.decision_zero_init_ffn

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class JevaDecisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.projection_dim
        self.num_heads = config.decision_num_heads
        self.head_dim = self.dim // self.num_heads
        self.depth_scale = 1 / math.sqrt(config.decision_depth)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.o_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.tau_set = JevaPositiveScalar(config.tau_set_init, config.min_temperature)
        ratio = min(max(config.alpha_over_tau_init, 1e-4), 1 - 1e-4)
        self.initial_gamma = math.log(ratio / (1 - ratio))
        self.gamma = nn.Parameter(torch.tensor([self.initial_gamma]))
        self.ffn_norm = nn.RMSNorm(self.dim, eps=1e-6) if config.decision_ffn_ratio > 0 else None
        self.ffn = JevaSwiGLU(config) if config.decision_ffn_ratio > 0 else None

    def forward(self, z, options, option_mask):
        batch_size, num_options = options.shape[:2]
        keys = self.k_proj(options).reshape(batch_size, num_options, self.num_heads, self.head_dim)
        values = self.v_proj(options).reshape(batch_size, num_options, self.num_heads, self.head_dim)
        keys = F.normalize(keys.float(), dim=-1)
        queries = self.q_proj(z).reshape(batch_size, self.num_heads, self.head_dim)
        queries = F.normalize(queries.float(), dim=-1)
        logits = torch.einsum("bhd,bkhd->bhk", queries, keys.float())
        attention = _log_softmax(
            (logits / self.tau_set().float()).masked_fill(~option_mask[:, None], -torch.inf)
        ).exp()
        attention = torch.where(option_mask[:, None], attention, torch.zeros_like(attention))
        message = torch.einsum("bhk,bkhd->bhd", attention, values.float()).reshape(batch_size, self.dim)
        message = self.o_proj(message.to(self.o_proj.weight.dtype)).to(z.dtype)
        alpha = self.tau_set() * torch.sigmoid(self.gamma) * self.depth_scale
        z = z + alpha.to(z.dtype) * message
        if self.ffn is not None:
            z = z + self.depth_scale * self.ffn(self.ffn_norm(z.to(self.ffn_norm.weight.dtype))).to(z.dtype)
        return z


class JevaDecisionStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.normalize_initial_state = config.normalize_state
        self.blocks = nn.ModuleList([JevaDecisionBlock(config) for _ in range(config.decision_depth)])
        self.final_norm = nn.RMSNorm(config.projection_dim, eps=1e-6) if config.decision_final_norm else nn.Identity()
        self.tau_dec = JevaPositiveScalar(config.tau_dec_init, config.min_temperature)
        self.tau_cov = JevaPositiveScalar(config.tau_cov_init, config.min_temperature)
        self.initial_coverage_bias = config.coverage_bias_init
        self.coverage_bias = nn.Parameter(torch.tensor([float(config.coverage_bias_init)]))

    def forward(self, state, options, option_mask):
        options = F.normalize(options, dim=-1)
        z = F.normalize(state, dim=-1) if self.normalize_initial_state else state
        for block in self.blocks:
            z = block(z, options, option_mask)
        scores = torch.einsum("bd,bkd->bk", self.final_norm(z).float(), options.float())
        masked_scores = scores.masked_fill(~option_mask, -torch.inf)
        log_q = _log_softmax((scores / self.tau_dec().float()).masked_fill(~option_mask, -torch.inf))
        coverage_logit = (masked_scores.max(dim=-1).values - self.coverage_bias.float()) / self.tau_cov().float()
        return log_q, coverage_logit


@dataclass
class JevaOutput(ModelOutput):
    """Conditional candidate log probabilities, coverage logits, and optional MoE auxiliary loss."""

    log_q: torch.Tensor | None = None
    coverage_logit: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


class JevaModel(PreTrainedModel):
    """
    Shared text encoder with a matrix decision stack for Choice, Noul, and Score tasks.

    The encoder is loaded with [`~transformers.AutoModel`]; no vocabulary projection is constructed. Use
    `from_backbone_pretrained` to initialize a new decision head, `from_jeva_checkpoint` to import a Jeva checkpoint,
    and [`~transformers.PreTrainedModel.from_pretrained`] to reload a checkpoint saved by this class.

    Args:
        config ([`experimental.jeva.JevaModelConfig`]):
            Encoder and decision-stack configuration.
        backbone ([`~transformers.PreTrainedModel`], *optional*):
            An already loaded shared encoder.
    """

    config_class = JevaModelConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _no_split_modules = ["JevaDecisionBlock"]
    _keep_in_fp32_modules = ["context_proj", "option_proj", "core"]
    _keep_in_fp32_modules_strict = ["context_proj", "option_proj", "core"]

    def __init__(self, config, backbone=None):
        super().__init__(config)
        self.backbone = backbone if backbone is not None else AutoModel.from_config(config.backbone_config)
        if config.adapter_config is not None:
            from peft import get_peft_config, get_peft_model

            self.backbone = get_peft_model(self.backbone, get_peft_config(config.adapter_config))
        self.context_proj = nn.Linear(config.backbone_config.hidden_size, config.projection_dim, bias=False)
        self.option_proj = nn.Linear(config.backbone_config.hidden_size, config.projection_dim, bias=False)
        self.core = JevaDecisionStack(config)
        if is_deepspeed_zero3_enabled():
            self._keep_in_fp32_modules = []
            self._keep_in_fp32_modules_strict = []
            for module in (self.context_proj, self.option_proj, self.core):
                module.to(dtype=self.backbone.dtype)
        self.freeze_backbone = False
        self.router_aux_loss_coef = 0.0
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, JevaPositiveScalar):
            nn.init.constant_(module.raw, module.initial_value)
        elif isinstance(module, JevaDecisionBlock):
            for projection in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
                nn.init.eye_(projection.weight)
            nn.init.constant_(module.gamma, module.initial_gamma)
        elif isinstance(module, JevaSwiGLU) and module.zero_init_output:
            nn.init.zeros_(module.down_proj.weight)
        elif isinstance(module, JevaDecisionStack):
            nn.init.constant_(module.coverage_bias, module.initial_coverage_bias)

    @classmethod
    def from_backbone_pretrained(cls, backbone_name_or_path, *, model_config=None, **kwargs):
        """
        Load a pretrained encoder and initialize the decision head.

        Args:
            backbone_name_or_path (`str`):
                Encoder identifier or local directory.
            model_config (`dict`, *optional*):
                Decision-stack arguments passed to [`experimental.jeva.JevaModelConfig`].
            **kwargs:
                Arguments forwarded to [`~transformers.AutoModel.from_pretrained`].
        """
        if "distributed_config" in kwargs or "tp_plan" in kwargs:
            raise ValueError("Save a JevaModel first, then use JevaModel.from_pretrained with distributed_config")
        backbone = AutoModel.from_pretrained(backbone_name_or_path, **kwargs)
        if backbone.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
            backbone = backbone.language_model
        config = JevaModelConfig(
            backbone_config=backbone.config,
            backbone_name_or_path=str(backbone_name_or_path),
            **(model_config or {}),
        )
        return cls(config, backbone=backbone)

    @classmethod
    def from_jeva_checkpoint(cls, checkpoint, *, revision=None, local_files_only=False, **kwargs):
        """
        Import a checkpoint containing `config.yaml`, `head.pt`, and an optional encoder adapter.

        Args:
            checkpoint (`str`):
                Local directory or Hub model identifier.
            revision (`str`, *optional*):
                Hub revision to download.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Load only locally available files.
            **kwargs:
                Encoder loading arguments forwarded to [`~transformers.AutoModel.from_pretrained`].
        """
        directory = Path(checkpoint)
        if not directory.is_dir():
            directory = Path(snapshot_download(checkpoint, revision=revision, local_files_only=local_files_only))
        with (directory / "config.yaml").open() as f:
            config = yaml.safe_load(f)["model"]
        if config.get("decision_arch", "matrix_stack") != "matrix_stack":
            raise ValueError("Jeva requires decision_arch=matrix_stack")
        backbone_path = config["backbone"]
        if (directory / backbone_path).exists():
            backbone_path = str(directory / backbone_path)
        if (directory / "backbone_full").is_dir():
            backbone_path = str(directory / "backbone_full")
        fields = (
            "projection_dim",
            "decision_depth",
            "decision_num_heads",
            "decision_ffn_ratio",
            "decision_dropout",
            "decision_zero_init_ffn",
            "decision_final_norm",
            "normalize_state",
            "pooling",
            "tau_set_init",
            "tau_dec_init",
            "tau_cov_init",
            "alpha_over_tau_init",
            "coverage_bias_init",
            "min_temperature",
        )
        model = cls.from_backbone_pretrained(
            backbone_path,
            model_config={key: config[key] for key in fields if key in config},
            local_files_only=local_files_only,
            **kwargs,
        )
        adapter_path = config.get("adapter_path")
        if (directory / "backbone_adapter").is_dir():
            adapter_path = str(directory / "backbone_adapter")
        elif adapter_path and (directory / adapter_path).exists():
            adapter_path = str(directory / adapter_path)
        if adapter_path:
            from peft import PeftModel

            model.backbone = PeftModel.from_pretrained(model.backbone, adapter_path, is_trainable=True)
            model.config.adapter_config = model.backbone.peft_config["default"].to_dict()
        head = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
        for name, module in (
            ("context_proj", model.context_proj),
            ("option_proj", model.option_proj),
            ("core", model.core),
        ):
            module.load_state_dict(
                {key: value.reshape(1) if value.ndim == 0 else value for key, value in head[name].items()}
            )
        return model

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode(self, inputs, cp_group=None, loss_mask=None):
        encoder_inputs = {key: value for key, value in inputs.items() if key != "sequence_lengths"}
        if self.router_aux_loss_coef:
            encoder_inputs["output_router_logits"] = True
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            outputs = self.backbone(**encoder_inputs, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"]
        positions = inputs.get("position_ids", torch.arange(mask.shape[1], device=mask.device)[None, :])
        lengths = inputs.get("sequence_lengths", mask.sum(-1))
        if self.config.pooling == "last":
            pool_mask = positions == lengths[:, None] - 1
        elif self.config.pooling == "cls":
            pool_mask = positions == 0
        else:
            pool_mask = mask.bool()
        pooled = (hidden * pool_mask.unsqueeze(-1).to(hidden.dtype)).sum(1)
        if cp_group is not None:
            pooled = all_reduce(pooled, group=cp_group)
        if self.config.pooling == "mean":
            pooled = pooled / lengths[:, None].clamp(min=1)

        aux_loss = hidden.new_zeros((), dtype=torch.float32)
        aux_tokens = mask.sum()
        if self.router_aux_loss_coef:
            from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

            router_logits = outputs.router_logits
            aux_mask = mask if loss_mask is None else mask * loss_mask[:, None]
            if cp_group is not None:
                router_logits = tuple(torch.cat(all_gather(logits, group=cp_group)) for logits in router_logits)
                aux_mask = torch.cat(all_gather(aux_mask, group=cp_group))
            aux_tokens = aux_mask.sum()
            if aux_tokens > 0:
                text_config = self.config.get_text_config()
                aux_loss = load_balancing_loss_func(
                    router_logits, text_config.num_experts, text_config.num_experts_per_tok, aux_mask
                )
            else:
                aux_loss = sum(logits.sum() for logits in router_logits) * 0.0
        return pooled, aux_loss, aux_tokens

    def forward(
        self,
        state_inputs,
        option_inputs,
        option_counts,
        option_mask,
        option_chunk_size=None,
        cp_group=None,
    ):
        """
        Encode states and options, returning conditional log probabilities and coverage logits.

        `state_inputs` and `option_inputs` contain tokenizer tensors. Options are flattened in example order;
        `option_counts` and `option_mask` describe the candidate sets. No inference cache is used.
        """
        state, aux_loss, aux_tokens = self._encode(state_inputs, cp_group)
        weighted_aux_loss = aux_loss * aux_tokens
        chunk_size = option_chunk_size or option_inputs["input_ids"].shape[0]
        option_embeddings = []
        real_options = option_counts.sum()
        for start in range(0, option_inputs["input_ids"].shape[0], chunk_size):
            chunk = {key: value[start : start + chunk_size] for key, value in option_inputs.items()}
            loss_mask = (
                torch.arange(start, start + chunk["input_ids"].shape[0], device=option_counts.device) < real_options
            )
            embeddings, chunk_aux_loss, chunk_aux_tokens = self._encode(chunk, cp_group, loss_mask)
            option_embeddings.append(embeddings)
            weighted_aux_loss = weighted_aux_loss + chunk_aux_loss * chunk_aux_tokens
            aux_tokens = aux_tokens + chunk_aux_tokens
        flat_options = torch.cat(option_embeddings)
        offsets = option_counts.cumsum(0) - option_counts
        indices = offsets[:, None] + torch.arange(option_mask.shape[1], device=option_counts.device)[None, :]
        options = flat_options[indices.clamp(max=flat_options.shape[0] - 1)]
        options = options.masked_fill(~option_mask[:, :, None], 0)
        with torch.autocast(device_type=state.device.type, enabled=False):
            log_q, coverage_logit = self.core(
                self.context_proj(state.to(self.context_proj.weight.dtype)),
                self.option_proj(options.to(self.option_proj.weight.dtype)),
                option_mask,
            )
        return JevaOutput(
            log_q=log_q,
            coverage_logit=coverage_logit,
            aux_loss=weighted_aux_loss / aux_tokens.clamp(min=1) if self.router_aux_loss_coef else None,
        )
