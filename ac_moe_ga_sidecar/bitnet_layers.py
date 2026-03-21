"""
BitNet-inspired low-bit layers for AC-MoE-GA Sidecar.

Implements native low-bit design principles:
- Dense core uses ternary/low-bit operations
- Repetitive compute paths are optimized for CPU/edge
- Fragile decision logic stays higher precision
- Memory movement minimized through shared projections

Key design decisions:
1. Ternary (±1, 0) for dense core projections
2. 8-bit quantization for encoder outputs
3. Higher precision (16-bit) for uncertainty/calibration
4. Shared projections to reduce memory traffic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BitNetConfig:
    """Configuration for BitNet-inspired low-bit design."""
    # Core architecture
    use_ternary_core: bool = True  # Use ternary (±1, 0) for dense core
    use_low_bit_encoders: bool = True  # Use 8-bit encoders
    shared_projections: bool = True  # Share projections across heads
    
    # Precision settings
    core_precision: int = 8  # 8-bit for dense core
    fragile_precision: int = 16  # 16-bit for uncertainty/calibration
    
    # Quantization settings
    quantize_activations: bool = True
    quantize_weights: bool = True
    
    # Training settings
    quantization_aware_training: bool = True
    freeze_scaling_factors: bool = False


class TernaryLinear(nn.Module):
    """Ternary linear layer (±1, 0) for low-bit dense core.
    
    Implements the BitNet approach: learn scale factors, use ternary weights.
    This reduces memory footprint and enables efficient CPU inference.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable scale factor (per output channel)
        self.scale = nn.Parameter(torch.ones(out_features))
        
        # Ternary weights (will be converted to ±1, 0 during forward)
        # We store as float for training, convert to ternary during inference
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize weights using BitNet-style initialization."""
        # Use small initial values to encourage sparsity
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def get_ternary_weights(self) -> torch.Tensor:
        """Convert weights to ternary (±1, 0)."""
        # Sign function with dead zone for sparsity
        threshold = 0.1
        ternary = torch.sign(self.weight)
        ternary[torch.abs(self.weight) < threshold] = 0.0
        return ternary
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary weights and learned scaling."""
        # Get ternary weights
        ternary_w = self.get_ternary_weights()
        
        # Apply scaling (per-channel)
        # Scale each output channel by its learned factor
        output = F.linear(x, ternary_w, self.bias)
        
        # Apply per-channel scaling
        output = output * self.scale.view(1, -1)
        
        return output


class LowBitEncoder(nn.Module):
    """Low-bit encoder with quantization support.
    
    Uses 8-bit quantization for encoder outputs to reduce memory traffic.
    Designed for CPU-friendly inference.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 use_ternary: bool = True, 
                 quantize_output: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_ternary = use_ternary
        self.quantize_output = quantize_output
        
        # Main projection
        if use_ternary:
            self.proj = TernaryLinear(input_dim, output_dim)
        else:
            self.proj = nn.Linear(input_dim, output_dim)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(output_dim)
        
        # Quantization scale (learned)
        self.quant_scale = nn.Parameter(torch.ones(1) * 10.0)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to low-bit representation."""
        if not self.quantize_output:
            return x
        
        # Quantize to 8-bit range
        x = torch.tanh(x / self.quant_scale)  # Squash to [-1, 1]
        x = torch.round(x * 127.0) / 127.0  # 8-bit quantization
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantization."""
        x = self.proj(x)
        x = self.norm(x)
        x = self.quantize(x)
        return x


class SharedProjection(nn.Module):
    """Shared projection layer for reducing memory traffic.
    
    Implements the BitNet insight: shared projections reduce memory movement
    more than they reduce expressivity for similar tasks.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Single shared projection
        self.proj = TernaryLinear(input_dim, output_dim * num_heads)
        
        # Per-head scaling
        self.head_scales = nn.Parameter(torch.ones(num_heads, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with shared projection and per-head scaling."""
        # Shared projection
        x = self.proj(x)  # (batch, output_dim * num_heads)
        
        # Reshape to per-head
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_heads, self.output_dim)
        
        # Apply per-head scaling
        x = x * self.head_scales
        
        return x  # (batch, num_heads, output_dim)


class QuantizationAwareModule(nn.Module):
    """Base class for quantization-aware training.
    
    Provides utilities for quantization-aware training with BitNet-style
    scaling and quantization.
    """
    
    def __init__(self, bit_width: int = 8):
        super().__init__()
        self.bit_width = bit_width
        self.quantize = True
        self._scale = nn.Parameter(torch.ones(1))
    
    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified bit width."""
        if not self.quantize:
            return x
        
        # Dynamic quantization
        max_val = torch.max(torch.abs(x))
        scale = max_val / (2 ** (self.bit_width - 1) - 1)
        
        # Quantize
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, -(2 ** (self.bit_width - 1)), 2 ** (self.bit_width - 1) - 1)
        
        # Dequantize
        x_dq = x_q * scale
        
        return x_dq
    
    def set_quantize(self, quantize: bool):
        """Enable/disable quantization."""
        self.quantize = quantize


class LowBitMLP(nn.Module):
    """Low-bit MLP for dense core.
    
    Implements BitNet-style dense core with ternary weights and quantization.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, use_ternary: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # First layer
        if use_ternary:
            self.layers = nn.ModuleList([
                TernaryLinear(input_dim, hidden_dim),
                TernaryLinear(hidden_dim, output_dim)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim),
                nn.Linear(hidden_dim, output_dim)
            ])
        
        # Norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) if use_ternary else nn.LayerNorm(hidden_dim)
            for _ in range(num_layers - 1)
        ])
        
        # Output norm
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through low-bit MLP."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i < len(self.layers) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        
        x = self.output_norm(x)
        return x


# Utility functions for BitNet operations

def ternary_quantize(x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Quantize tensor to ternary (±1, 0)."""
    ternary = torch.sign(x)
    ternary[torch.abs(x) < threshold] = 0.0
    return ternary


def dynamic_quantize(x: torch.Tensor, bit_width: int = 8) -> torch.Tensor:
    """Dynamic quantization to specified bit width."""
    max_val = torch.max(torch.abs(x))
    num_levels = 2 ** (bit_width - 1) - 1
    scale = max_val / num_levels
    
    # Quantize
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, -num_levels, num_levels)
    
    # Dequantize
    x_dq = x_q * scale
    
    return x_dq


def compute_ternary_flops(in_features: int, out_features: int, batch_size: int) -> int:
    """Estimate FLOPs for ternary linear layer.
    
    Ternary operations can be implemented with shifts and adds,
    reducing FLOPs compared to full-precision multiply-accumulates.
    """
    # For ternary, each weight is ±1 or 0
    # Can be implemented as: sum(sign(w) * x) = sum(x) - 2*sum(x where w=-1)
    # This is roughly 2x faster than full-precision
    macs = in_features * out_features * batch_size
    # Ternary reduces to ~2 operations per MAC
    return 2 * macs
