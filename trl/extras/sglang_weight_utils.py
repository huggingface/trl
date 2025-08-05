# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

"""
SGLang Weight Update Utilities - Slime-style batched weight updates with bucketing and memory management.
"""

import gc
import logging
import socket
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

from accelerate.utils import is_peft_model


logger = logging.getLogger(__name__)


@dataclass
class ParamInfo:
    """Parameter information for distributed weight updates."""
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict  # Contains tensor parallelism and other attributes
    size: int  # Parameter size in bytes
    src_rank: int  # Source rank that owns this parameter


class SGLangWeightUpdater:
    """
    Slime-style batched weight updater for SGLang engines with memory management and bucketing.
    
    This class implements sophisticated parameter bucketing and distributed weight updates
    following the patterns from the slime framework for optimal performance.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        sglang_mode: str,
        sglang_client=None,
        sglang_engine=None,
        accelerator=None,
        update_weight_buffer_size: int = 512 * 1024**2,  # 512MB default
    ):
        self.model = model
        self.sglang_mode = sglang_mode
        self.sglang_client = sglang_client
        self.sglang_engine = sglang_engine
        self.accelerator = accelerator
        self.update_weight_buffer_size = update_weight_buffer_size
        
        # Initialize distributed groups if needed
        self._init_distributed_groups()
    
    def _init_distributed_groups(self):
        """Initialize distributed process groups for weight updates."""
        if self.accelerator and self.accelerator.is_main_process:
            self._is_main_process = True
            self._group_name = "sglang_weight_sync"
        else:
            self._is_main_process = False
    
    def get_param_infos(self, model: torch.nn.Module) -> list[ParamInfo]:
        """
        Extract parameter information from the model.
        
        Args:
            model: The model to extract parameters from
            
        Returns:
            List of ParamInfo objects containing parameter metadata
        """
        param_infos = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Calculate parameter size in bytes
            param_size = param.numel() * param.element_size()
            
            # Extract tensor parallelism attributes if available
            attrs = {}
            if hasattr(param, 'tensor_model_parallel'):
                attrs['tensor_model_parallel'] = param.tensor_model_parallel
            if hasattr(param, 'partition_dim'):
                attrs['partition_dim'] = param.partition_dim
            if hasattr(param, 'partition_stride'):
                attrs['partition_stride'] = param.partition_stride
            
            # Determine source rank (simplified - in full implementation would handle TP/PP)
            src_rank = 0 if self.accelerator is None else self.accelerator.process_index
            
            param_info = ParamInfo(
                name=name,
                dtype=param.dtype,
                shape=param.shape,
                attrs=attrs,
                size=param_size,
                src_rank=src_rank
            )
            param_infos.append(param_info)
        
        return param_infos
    
    def get_param_info_buckets(self, param_infos: list[ParamInfo]) -> list[list[ParamInfo]]:
        """
        Group parameters into buckets based on memory constraints.
        
        Args:
            param_infos: List of parameter information
            
        Returns:
            List of parameter buckets, each respecting the buffer size limit
        """
        param_info_buckets = [[]]
        buffer_size = 0
        oversized_params = []
        
        for info in param_infos:
            # Calculate effective parameter size (accounting for tensor parallelism)
            tp_size = 1
            if hasattr(self.accelerator, 'num_processes'):
                tp_size = self.accelerator.num_processes
            
            # Handle expert parameters if present (MoE models)
            if '.experts.' in info.name:
                # For expert parameters, we might have different TP size
                effective_param_size = info.size * tp_size
            else:
                effective_param_size = info.size * tp_size
            
            # If a single parameter exceeds the buffer size, handle it separately
            if effective_param_size > self.update_weight_buffer_size:
                oversized_params.append(info)
                logger.warning(f"Parameter {info.name} ({effective_param_size / (1024**2):.2f}MB) exceeds buffer size ({self.update_weight_buffer_size / (1024**2):.2f}MB), will be processed individually")
                continue
            
            # Check if adding this parameter would exceed the buffer size
            if buffer_size + effective_param_size > self.update_weight_buffer_size and param_info_buckets[-1]:
                # Start a new bucket
                param_info_buckets.append([])
                buffer_size = 0
            
            param_info_buckets[-1].append(info)
            buffer_size += effective_param_size
        
        # Add oversized parameters as individual buckets
        for oversized_param in oversized_params:
            param_info_buckets.append([oversized_param])
        
        # Remove empty buckets
        param_info_buckets = [bucket for bucket in param_info_buckets if bucket]
        
        logger.info(f"Created {len(param_info_buckets)} parameter buckets with buffer size {self.update_weight_buffer_size / (1024**2):.2f}MB")
        if oversized_params:
            logger.info(f"Found {len(oversized_params)} oversized parameters that will be processed individually")
        
        return param_info_buckets
    
    def _fix_param_name_for_sglang(self, name: str, extra_prefixes: Optional[list[str]] = None) -> str:
        """Fix parameter names for SGLang compatibility."""
        extra_prefixes = extra_prefixes or []
        prefixes_to_remove = ["_checkpoint_wrapped_module."] + extra_prefixes
        
        for prefix in prefixes_to_remove:
            name = name.replace(prefix, "")
        
        return name
    
    def _update_bucket_weights_server_mode(self, bucket_params: list[tuple[str, torch.Tensor]]) -> None:
        """
        Update weights for a bucket of parameters in server mode.
        
        Args:
            bucket_params: List of (name, parameter) tuples for this bucket
        """
        if not self.accelerator.is_main_process:
            return
        
        names = [name for name, _ in bucket_params]
        dtypes = [str(param.dtype) for _, param in bucket_params]
        shapes = [list(param.shape) for _, param in bucket_params]
        
        # Use SGLang client's batch update API
        url = f"{self.sglang_client.base_url}/update_weights/"
        response = self.sglang_client.session.post(
            url,
            json={
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": self._group_name,
                "flush_cache": False,  # Don't flush cache for each bucket
            },
        )
        
        if response.status_code != 200:
            raise Exception(f"SGLang bucket weight update failed: {response.status_code}, {response.text}")
        
        logger.debug(f"Updated bucket with {len(names)} parameters in server mode")
    
    def _update_bucket_weights_colocate_mode(self, bucket_params: list[tuple[str, torch.Tensor]]) -> None:
        """
        Update weights for a bucket of parameters in colocate mode.
        
        Args:
            bucket_params: List of (name, parameter) tuples for this bucket
        """
        names = [name for name, _ in bucket_params]
        dtypes = [str(param.dtype) for _, param in bucket_params]
        shapes = [list(param.shape) for _, param in bucket_params]
        
        # Single NCCL operation for all parameters in the bucket
        try:
            self.sglang_engine.update_weights_from_distributed(names, dtypes, shapes, self._group_name)
        except Exception as e:
            logger.warning(f"SGLang weight update failed: {e}")
            logger.warning("Falling back to individual parameter updates")
            # Fallback to individual updates if batch update fails
            for name, dtype, shape in zip(names, dtypes, shapes):
                try:
                    self.sglang_engine.update_weights_from_distributed([name], [dtype], [shape], self._group_name)
                except Exception as e2:
                    logger.error(f"Failed to update parameter {name}: {e2}")
        
        logger.debug(f"Updated bucket with {len(names)} parameters in colocate mode")
    
    def _process_peft_parameters_bucketed(self, model: torch.nn.Module, gather_if_zero3) -> None:
        """Process PEFT parameters using bucketed updates."""
        # Collect all PEFT parameters first
        peft_params = []
        param_infos = []
        
        for name, param in model.named_parameters():
            # Apply PEFT parameter name transformations
            original_name = name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            
            # Skip certain PEFT parameters
            if "lora_dropout" in name or "modules_to_save.default.lm_head" in name:
                continue
            if "original_module" in name:
                continue
            if hasattr(model, 'prefix') and model.prefix in name:
                continue
                
            name = self._fix_param_name_for_sglang(name, extra_prefixes=["modules_to_save.default."])
            peft_params.append((name, param))
            
            # Create param info for bucketing
            param_info = ParamInfo(
                name=name,
                dtype=param.dtype,
                shape=param.shape,
                attrs={},
                size=param.numel() * param.element_size(),
                src_rank=0
            )
            param_infos.append(param_info)
        
        if not peft_params:
            return
        
        # Create buckets for PEFT parameters
        param_buckets = self.get_param_info_buckets(param_infos)
        
        # Process each bucket
        with tqdm(total=len(param_buckets), desc="Updating PEFT parameter buckets") as pbar:
            for bucket_infos in param_buckets:
                bucket_params = []
                for param_info in bucket_infos:
                    # Find the corresponding parameter
                    for name, param in peft_params:
                        if name == param_info.name:
                            bucket_params.append((name, param))
                            break
                
                if bucket_params:
                    if self.sglang_mode == "server":
                        self._update_bucket_weights_server_mode(bucket_params)
                    elif self.sglang_mode == "colocate":
                        self._update_bucket_weights_colocate_mode(bucket_params)
                
                pbar.update(1)
    
    def _process_regular_parameters_bucketed(self, model: torch.nn.Module, gather_if_zero3) -> None:
        """Process regular parameters using bucketed updates."""
        # Collect all regular parameters first
        regular_params = []
        param_infos = []
        
        for name, param in model.named_parameters():
            name = self._fix_param_name_for_sglang(name)
            regular_params.append((name, param))
            
            # Create param info for bucketing
            param_info = ParamInfo(
                name=name,
                dtype=param.dtype,
                shape=param.shape,
                attrs={},
                size=param.numel() * param.element_size(),
                src_rank=0
            )
            param_infos.append(param_info)
        
        if not regular_params:
            return
        
        # Create buckets for regular parameters
        param_buckets = self.get_param_info_buckets(param_infos)
        
        # Process each bucket with parameter gathering
        with tqdm(total=len(param_buckets), desc="Updating regular parameter buckets") as pbar:
            for bucket_infos in param_buckets:
                bucket_params = []
                params_to_gather = []
                
                for param_info in bucket_infos:
                    # Find the corresponding parameter
                    for name, param in regular_params:
                        if name == param_info.name:
                            bucket_params.append((name, param))
                            params_to_gather.append(param)
                            break
                
                if bucket_params:
                    # Gather all parameters in this bucket at once
                    with gather_if_zero3(params_to_gather):
                        if self.sglang_mode == "server":
                            self._update_bucket_weights_server_mode(bucket_params)
                        elif self.sglang_mode == "colocate":
                            self._update_bucket_weights_colocate_mode(bucket_params)
                
                pbar.update(1)
    
    def _flush_sglang_cache(self):
        """Flush SGLang cache after all weight updates."""
        if self.sglang_mode == "server" and self.accelerator.is_main_process:
            self.sglang_client.flush_cache()
        elif self.sglang_mode == "colocate":
            self.sglang_engine.reset_prefix_cache()
    
    def update_model_weights(self, deepspeed_plugin=None) -> None:
        """
        Update SGLang model weights using slime-style bucketed approach.
        
        This is the main entry point that handles the complete weight update process
        with memory-aware bucketing and proper distributed coordination.
        """
        logger.info("Starting slime-style bucketed weight update")
        start_time = time.time()
        
        # Setup gathering context for DeepSpeed ZeRO-3
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext
        
        # Clear GPU memory before starting
        self._clear_gpu_memory()
        
        if is_peft_model(self.model):
            # Handle PEFT models with adapter merging
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()
                
                # Process PEFT parameters with bucketing
                if hasattr(self.accelerator.state, 'fsdp_plugin') and self.accelerator.state.fsdp_plugin is not None:
                    # FSDP handling would go here - simplified for now
                    self._process_peft_parameters_bucketed(self.model, gather_if_zero3)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    self._process_peft_parameters_bucketed(self.model, gather_if_zero3)
                
                # Unmerge adapters after update
                self.model.unmerge_adapter()
        else:
            # Handle regular models without PEFT
            if hasattr(self.accelerator.state, 'fsdp_plugin') and self.accelerator.state.fsdp_plugin is not None:
                # FSDP handling would go here - simplified for now
                self._process_regular_parameters_bucketed(self.model, gather_if_zero3)
            else:
                # Regular parameter processing with bucketing
                self._process_regular_parameters_bucketed(self.model, gather_if_zero3)
        
        # Flush cache once at the end
        self._flush_sglang_cache()
        
        # Clear memory after updates
        self._clear_gpu_memory()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed slime-style weight update in {elapsed_time:.2f} seconds")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM issues."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> dict[str, Any]:
        """Get current GPU memory information."""
        if not torch.cuda.is_available():
            return {"gpu": "none", "total_GB": 0, "free_GB": 0, "used_GB": 0}
        
        free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
        return {
            "gpu": str(torch.cuda.current_device()),
            "total_GB": round(total / (1024**3), 2),
            "free_GB": round(free / (1024**3), 2),
            "used_GB": round((total - free) / (1024**3), 2),
        }