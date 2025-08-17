"""
Parameter Efficient Fine-Tuning (PEFT) implementation for CodeLLM.

This module provides LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning
of large language models with minimal parameter updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 8  # Rank of the low-rank matrices
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1  # Dropout probability
    target_modules: List[str] = None  # Target modules to apply LoRA
    bias: str = "none"  # Bias handling: "none", "all", "lora_only"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class LoRALayer(nn.Module):
    """LoRA layer implementation for linear transformations."""
    
    def __init__(self, in_features: int, out_features: int, config: LoRAConfig):
        super().__init__()
        self.config = config
        
        # Original weight (frozen)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        
        # LoRA components
        self.lora_A = nn.Parameter(torch.zeros(config.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))
        
        # Scaling factor
        self.scaling = config.lora_alpha / config.r
        
        # Dropout
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        
        # Bias handling
        if config.bias == "all":
            self.bias = nn.Parameter(torch.zeros(out_features))
        elif config.bias == "lora_only":
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
            self.lora_bias = None
        
        # Initialize LoRA weights
        self._init_lora_weights()
    
    def _init_lora_weights(self):
        """Initialize LoRA weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        if self.config.bias == "lora_only" and self.lora_bias is not None:
            nn.init.zeros_(self.lora_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Original forward pass
        output = F.linear(x, self.weight, self.bias)
        
        # LoRA forward pass
        lora_output = self.lora_dropout(x)
        lora_output = F.linear(lora_output, self.lora_A.t())
        lora_output = F.linear(lora_output, self.lora_B.t())
        lora_output = lora_output * self.scaling
        
        # Add LoRA bias if configured
        if self.config.bias == "lora_only" and self.lora_bias is not None:
            lora_output = lora_output + self.lora_bias
        
        return output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into the original weight matrix."""
        if self.config.bias == "lora_only" and self.lora_bias is not None:
            self.bias = nn.Parameter(self.lora_bias.data.clone())
        
        # Merge LoRA weights
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data += lora_weight
        
        # Clear LoRA weights
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()


class LoRALinear(nn.Module):
    """LoRA-enhanced linear layer that can replace standard linear layers."""
    
    def __init__(self, linear_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, config)
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through both original and LoRA layers."""
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to a model by replacing target modules with LoRA-enhanced versions.
    
    Args:
        model: The model to apply LoRA to
        config: LoRA configuration
    
    Returns:
        Model with LoRA applied to target modules
    """
    for name, module in model.named_modules():
        if any(target in name for target in config.target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA-enhanced version
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = model.get_submodule(parent_name) if parent_name else model
                
                lora_linear = LoRALinear(module, config)
                setattr(parent_module, child_name, lora_linear)
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.
    
    Args:
        model: Model with LoRA applied
    
    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend(module.parameters())
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """
    Save LoRA weights to a file.
    
    Args:
        model: Model with LoRA applied
        path: Path to save the weights
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            prefix = name + "."
            lora_state_dict[prefix + "lora_A"] = module.lora_A.data
            lora_state_dict[prefix + "lora_B"] = module.lora_B.data
            if module.lora_bias is not None:
                lora_state_dict[prefix + "lora_bias"] = module.lora_bias.data
    
    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA weights from a file.
    
    Args:
        model: Model with LoRA applied
        path: Path to load the weights from
    """
    lora_state_dict = torch.load(path, map_location='cpu')
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            prefix = name + "."
            if prefix + "lora_A" in lora_state_dict:
                module.lora_A.data = lora_state_dict[prefix + "lora_A"]
            if prefix + "lora_B" in lora_state_dict:
                module.lora_B.data = lora_state_dict[prefix + "lora_B"]
            if prefix + "lora_bias" in lora_state_dict and module.lora_bias is not None:
                module.lora_bias.data = lora_state_dict[prefix + "lora_bias"]


class PEFTModel(nn.Module):
    """Wrapper class for PEFT-enhanced models."""
    
    def __init__(self, base_model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Apply LoRA to the base model
        self.base_model = apply_lora_to_model(self.base_model, config)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the PEFT-enhanced model."""
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters (LoRA parameters only)."""
        return get_lora_parameters(self.base_model)
    
    def save_lora(self, path: str):
        """Save LoRA weights."""
        save_lora_weights(self.base_model, path)
    
    def load_lora(self, path: str):
        """Load LoRA weights."""
        load_lora_weights(self.base_model, path)


# Compatibility with Hugging Face PEFT library
def get_peft_model(model: nn.Module, config: LoRAConfig) -> PEFTModel:
    """
    Create a PEFT model from a base model and LoRA configuration.
    
    Args:
        model: Base model to apply PEFT to
        config: LoRA configuration
    
    Returns:
        PEFT-enhanced model
    """
    return PEFTModel(model, config)


# Configuration class compatible with Hugging Face PEFT
class LoraConfig:
    """Compatibility class for Hugging Face PEFT LoraConfig."""
    
    def __init__(self, r: int = 8, lora_alpha: int = 16, target_modules: List[str] = None,
                 lora_dropout: float = 0.1, bias: str = "none"):
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.lora_dropout = lora_dropout
        self.bias = bias
