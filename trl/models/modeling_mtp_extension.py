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

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


@dataclass
class CausalLMOutputWithMTP(CausalLMOutput):
    """
    Extended CausalLMOutput to include Multi-Token Prediction logits and loss.
    
    Args:
        mtp_logits (`List[torch.FloatTensor]` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores for each future token position (k predictions).
        mtp_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Multi-token prediction loss.
    """
    mtp_logits: Optional[List[torch.FloatTensor]] = None
    mtp_loss: Optional[torch.FloatTensor] = None


class MTPHeads(nn.Module):
    """
    Multi-Token Prediction heads that predict multiple future tokens in parallel.
    
    Args:
        config: Model configuration containing hidden_size and vocab_size.
        num_predictions (`int`): Number of future tokens to predict.
        head_type (`str`): Type of head architecture ('linear', 'ffn', 'mha_ffn', 'cnn', 'identical').

        dropout_prob (`float`): Dropout probability for regularization.
        num_layers (`int`): Number of layers in each MTP head (for multi-layer heads).
        init_strategy (`str`): Parameter initialization strategy ('default', 'kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', 'copy_lm_head').
        lm_head_module (`nn.Module`, optional): Reference to the original LM head for copying structure/weights.
    """
    
    def __init__(
        self,
        config,
        num_predictions: int = 2,
        head_type: str = "linear",
        dropout_prob: float = 0.1,
        num_layers: int = 1,
        init_strategy: str = "default",
        lm_head_module: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__()
        self.num_predictions = num_predictions
        self.head_type = head_type
        self.num_layers = num_layers
        self.init_strategy = init_strategy
        self.lm_head_module = lm_head_module
        
        # Get hidden size from config (handle different model architectures)
        hidden_size = self._get_hidden_size(config)
        vocab_size = config.vocab_size
        
        # Create dropout layer
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        
        # Create MTP heads based on type
        if head_type == "linear":
            self.heads = nn.ModuleList([
                self._create_multi_layer_linear(hidden_size, vocab_size, num_layers)
                for _ in range(num_predictions)
            ])
        elif head_type == "ffn":
            self.heads = nn.ModuleList([
                self._create_multi_layer_ffn(hidden_size, vocab_size, num_layers)
                for _ in range(num_predictions)
            ])
        elif head_type == "mha_ffn":
            self.heads = nn.ModuleList([
                MHAFFNHead(hidden_size, vocab_size, num_layers) for _ in range(num_predictions)
            ])
        elif head_type == "cnn":
            self.heads = nn.ModuleList([
                CNNHead(hidden_size, vocab_size, num_layers) for _ in range(num_predictions)
            ])
        elif head_type == "identical":
            self.heads = nn.ModuleList([
                self._create_identical_head(hidden_size, vocab_size, lm_head_module)
                for _ in range(num_predictions)
            ])
        else:
            raise ValueError(f"Unsupported head_type: {head_type}. Supported types: 'linear', 'ffn', 'mha_ffn', 'cnn', 'identical'")
        
        # Apply initialization strategy
        self._initialize_parameters()
    
    def _get_hidden_size(self, config):
        """Extract hidden size from different model configurations."""
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        elif hasattr(config, "word_embed_proj_dim"):
            return config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder:
            if hasattr(config, "decoder") and hasattr(config.decoder, "hidden_size"):
                return config.decoder.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size from model config")
    
    def _create_multi_layer_linear(self, hidden_size: int, vocab_size: int, num_layers: int) -> nn.Module:
        """Create multi-layer linear head."""
        if num_layers == 1:
            return nn.Linear(hidden_size, vocab_size)
        
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        layers.append(nn.Linear(hidden_size, vocab_size))
        return nn.Sequential(*layers)
    
    def _create_multi_layer_ffn(self, hidden_size: int, vocab_size: int, num_layers: int) -> nn.Module:
        """Create multi-layer FFN head."""
        if num_layers == 1:
            return nn.Sequential(
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.GELU(),
                nn.Linear(4 * hidden_size, vocab_size)
            )
        
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.GELU(),
                nn.Linear(4 * hidden_size, hidden_size),
                nn.Dropout(0.1),
            ])
        # Final layer
        layers.extend([
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, vocab_size)
        ])
        return nn.Sequential(*layers)
    
    def _create_identical_head(self, hidden_size: int, vocab_size: int, lm_head_module: Optional[nn.Module]) -> nn.Module:
        """Create head identical to the original LM head structure."""
        if lm_head_module is None:
            # Default to simple linear if no LM head provided
            return nn.Linear(hidden_size, vocab_size)
        
        # Clone the structure of the LM head
        if isinstance(lm_head_module, nn.Linear):
            return nn.Linear(lm_head_module.in_features, lm_head_module.out_features, bias=lm_head_module.bias is not None)
        elif isinstance(lm_head_module, nn.Sequential):
            # Deep copy the sequential structure
            layers = []
            for layer in lm_head_module:
                if isinstance(layer, nn.Linear):
                    layers.append(nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None))
                elif isinstance(layer, nn.LayerNorm):
                    layers.append(nn.LayerNorm(layer.normalized_shape, eps=layer.eps, elementwise_affine=layer.elementwise_affine))
                elif isinstance(layer, nn.ReLU):
                    layers.append(nn.ReLU())
                elif isinstance(layer, nn.GELU):
                    layers.append(nn.GELU())
                elif isinstance(layer, nn.Dropout):
                    layers.append(nn.Dropout(layer.p))
                else:
                    # For other layer types, try to create a similar layer
                    layers.append(type(layer)())
            return nn.Sequential(*layers)
        else:
            # For complex modules, create a simple linear layer as fallback
            return nn.Linear(hidden_size, vocab_size)
    
    def _initialize_parameters(self):
        """Initialize parameters based on the specified strategy."""
        if self.init_strategy == "default":
            # Use PyTorch's default initialization
            return
        elif self.init_strategy == "copy_lm_head" and self.lm_head_module is not None:
            self._copy_lm_head_parameters()
        else:
            # Apply specific initialization strategies
            for head in self.heads:
                self._apply_initialization_to_module(head, self.init_strategy)
    
    def _copy_lm_head_parameters(self):
        """Copy parameters from the original LM head to MTP heads."""
        if self.lm_head_module is None:
            return
        
        for head in self.heads:
            self._copy_module_parameters(self.lm_head_module, head)
    
    def _copy_module_parameters(self, source: nn.Module, target: nn.Module):
        """Copy parameters from source module to target module."""
        source_params = dict(source.named_parameters())
        target_params = dict(target.named_parameters())
        
        for name, param in target_params.items():
            if name in source_params and param.shape == source_params[name].shape:
                param.data.copy_(source_params[name].data)
    
    def _apply_initialization_to_module(self, module: nn.Module, strategy: str):
        """Apply initialization strategy to a module."""
        for name, param in module.named_parameters():
            if 'weight' in name:
                if strategy == "kaiming_uniform":
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif strategy == "kaiming_normal":
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                elif strategy == "xavier_uniform":
                    nn.init.xavier_uniform_(param)
                elif strategy == "xavier_normal":
                    nn.init.xavier_normal_(param)
            elif 'bias' in name and param is not None:
                nn.init.zeros_(param)
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through MTP heads.
        
        Args:
            hidden_states: Hidden states from the transformer backbone.
            
        Returns:
            List of logits tensors, one for each future token prediction.
        """
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Ensure proper dtype for numerical stability
        if hidden_states.dtype != self.heads[0][0].weight.dtype if isinstance(self.heads[0], nn.Sequential) else self.heads[0].weight.dtype:
            target_dtype = self.heads[0][0].weight.dtype if isinstance(self.heads[0], nn.Sequential) else self.heads[0].weight.dtype
            hidden_states = hidden_states.to(target_dtype)
        
        # Generate predictions for each future position
        mtp_logits = []
        for head in self.heads:
            logits = head(hidden_states)
            mtp_logits.append(logits)
        
        return mtp_logits
    



class MHAFFNHead(nn.Module):
    """Multi-Head Attention + FFN head for MTP with multi-layer support."""
    
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Create multiple transformer-like layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=min(8, hidden_size // 64),  # Adaptive number of heads
                    dropout=0.1,
                    batch_first=True
                ),
                'norm1': nn.LayerNorm(hidden_size),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, 4 * hidden_size),
                    nn.GELU(),
                    nn.Linear(4 * hidden_size, hidden_size)
                ),
                'norm2': nn.LayerNorm(hidden_size),
                'dropout': nn.Dropout(0.1)
            })
            self.layers.append(layer)
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pass through multiple transformer-like layers
        for layer in self.layers:
            # Self-attention with residual connection
            attn_output, _ = layer['attention'](hidden_states, hidden_states, hidden_states)
            hidden_states = layer['norm1'](hidden_states + layer['dropout'](attn_output))
            
            # FFN with residual connection
            ffn_output = layer['ffn'](hidden_states)
            hidden_states = layer['norm2'](hidden_states + layer['dropout'](ffn_output))
        
        # Output projection
        return self.output_projection(hidden_states)


class CNNHead(nn.Module):
    """1D CNN head for MTP with multi-layer support."""
    
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        
        # Create multiple CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'conv': nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                'norm': nn.LayerNorm(hidden_size),
                'activation': nn.GELU(),
                'dropout': nn.Dropout(0.1)
            })
            self.conv_layers.append(layer)
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, hidden_size)
        
        for layer in self.conv_layers:
            # Conv1d expects: (batch_size, hidden_size, seq_len)
            hidden_states_transposed = hidden_states.transpose(1, 2)
            conv_output = layer['conv'](hidden_states_transposed)
            conv_output = layer['activation'](conv_output)
            # Transpose back: (batch_size, seq_len, hidden_size)
            conv_output = conv_output.transpose(1, 2)
            
            # Apply normalization and dropout
            conv_output = layer['norm'](conv_output)
            conv_output = layer['dropout'](conv_output)
            
            # Residual connection
            hidden_states = hidden_states + conv_output
        
        return self.output_projection(hidden_states)


class MTPExtension:
    """
    Utility class to dynamically add/remove MTP functionality to/from models.
    """
    
    @staticmethod
    def add_mtp_to_model(
        model: PreTrainedModel,
        num_predictions: int = 2,
        head_type: str = "linear",
        dropout_prob: float = 0.1,
        num_layers: int = 1,
        init_strategy: str = "default",
        **kwargs
    ):
        """
        Dynamically add MTP functionality to a model.
        
        Args:
            model: The model to extend with MTP.
            num_predictions: Number of future tokens to predict.
            head_type: Type of MTP head architecture ('linear', 'ffn', 'mha_ffn', 'cnn', 'identical').
            dropout_prob: Dropout probability.
            num_layers: Number of layers in each MTP head.
            init_strategy: Parameter initialization strategy ('default', 'copy_lm_head', etc.).
        """
        if hasattr(model, 'mtp_heads'):
            return  # MTP already added
        
        # Add MTP configuration to model config
        if not hasattr(model.config, 'mtp_enabled'):
            model.config.mtp_enabled = True
            model.config.mtp_num_predictions = num_predictions
            model.config.mtp_head_type = head_type
            model.config.mtp_num_layers = num_layers
            model.config.mtp_init_strategy = init_strategy
        
        # Get reference to LM head for structure copying if needed
        lm_head_module = None
        if hasattr(model, 'lm_head'):
            lm_head_module = model.lm_head
        elif hasattr(model, 'head'):
            lm_head_module = model.head
        elif hasattr(model, 'output_layer'):
            lm_head_module = model.output_layer
        
        # Create and attach MTP heads
        model.mtp_heads = MTPHeads(
            config=model.config,
            num_predictions=num_predictions,
            head_type=head_type,
            dropout_prob=dropout_prob,
            num_layers=num_layers,
            init_strategy=init_strategy,
            lm_head_module=lm_head_module,
            **kwargs
        )
        
        # Store original forward method and replace with MTP-enabled version
        if not hasattr(model, '_original_forward'):
            model._original_forward = model.forward
            model.forward = lambda *args, **kwargs: mtp_forward(model, *args, **kwargs)
    
    @staticmethod
    def remove_mtp_from_model(model: PreTrainedModel):
        """
        Remove MTP functionality from a model and restore original behavior.
        
        Args:
            model: The model to remove MTP from.
        """
        if hasattr(model, '_original_forward'):
            model.forward = model._original_forward
            delattr(model, '_original_forward')
        
        if hasattr(model, 'mtp_heads'):
            delattr(model, 'mtp_heads')
        
        # Clean up config
        for attr in ['mtp_enabled', 'mtp_num_predictions', 'mtp_head_type', 'mtp_num_layers', 'mtp_init_strategy']:
            if hasattr(model.config, attr):
                delattr(model.config, attr)


def mtp_forward(model: PreTrainedModel, *args, **kwargs):
    """
    Modified forward pass that includes MTP predictions.
    
    This function replaces the model's original forward method when MTP is enabled.
    """
    # Call original forward method
    outputs = model._original_forward(*args, **kwargs)
    
    # Only add MTP during training
    if not model.training or not hasattr(model, 'mtp_heads'):
        return outputs
    
    # Get hidden states from the last layer
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        # Use the last layer's hidden states
        hidden_states = outputs.hidden_states[-1]
    elif hasattr(outputs, 'last_hidden_state'):
        hidden_states = outputs.last_hidden_state
    else:
        # Fallback: try to get hidden states from logits
        # This is a rough approximation and may not work for all models
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            # For GPT-style models
            hidden_states = model.transformer.ln_f(outputs.logits)
        else:
            # Skip MTP if we can't get hidden states
            return outputs
    
    # Generate MTP predictions
    mtp_logits = model.mtp_heads(hidden_states)
    
    # Create extended output
    return CausalLMOutputWithMTP(
        loss=outputs.loss,
        logits=outputs.logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        mtp_logits=mtp_logits,
        mtp_loss=None,  # Will be computed in the trainer
    )
