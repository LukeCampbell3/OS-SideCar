"""
BitNet-inspired quantization-aware training for AC-MoE-GA Sidecar.

Implements training-time low-bit awareness:
- Quantization-aware training for dense core
- Low-bit-friendly activation ranges
- Training with same approximate arithmetic as deployment
- Ternary regularization to encourage sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import SidecarConfig, TrainingConfig
from .bitnet_model import BitNetACMoEGAModel, BitNetModelState
from .bitnet_layers import TernaryLinear, ternary_quantize, dynamic_quantize


@dataclass
class BitNetTrainingState:
    """State for BitNet training."""
    step: int = 0
    ternary_sparsity: float = 0.0
    quantization_error: float = 0.0
    scaling_factor_mean: float = 1.0


class BitNetQuantizationAwareTrainer:
    """Trainer with quantization-aware training for BitNet-style models.
    
    Key features:
    - Quantization-aware training for dense core
    - Ternary regularization to encourage sparsity
    - Low-bit-friendly activation ranges
    - Training with same approximate arithmetic as deployment
    """
    
    def __init__(self, model: BitNetACMoEGAModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.state = BitNetTrainingState()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.sequence_length
        )
    
    def compute_ternary_regularization(self) -> torch.Tensor:
        """Compute ternary regularization loss.
        
        Encourages weights to be exactly ±1 or 0, reducing model size.
        """
        reg_loss = torch.tensor(0.0, device=self.model.device)
        
        for module in self.model.modules():
            if isinstance(module, TernaryLinear):
                # Encourage weights to be near 0, 1, or -1
                weights = module.weight
                reg_loss = reg_loss + torch.mean(torch.abs(weights) * (1 - torch.abs(weights)) ** 2)
        
        return reg_loss * self.config.ternary_regularization_weight
    
    def compute_quantization_loss(self, original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """Compute quantization loss (distillation from full-precision to low-bit)."""
        return F.mse_loss(original, quantized)
    
    def quantize_model_weights(self):
        """Quantize model weights for inference."""
        for module in self.model.modules():
            if isinstance(module, TernaryLinear):
                # Convert to ternary
                ternary_w = ternary_quantize(module.weight)
                module.weight.data = ternary_w
    
    def forward_with_quantization(self, input_tensors: Dict[str, torch.Tensor],
                                   state: Optional[BitNetModelState] = None) -> Tuple:
        """Forward pass with quantization simulation.
        
        Simulates quantization during training to prepare for low-bit inference.
        """
        # Quantize inputs
        quantized_inputs = {}
        for key, value in input_tensors.items():
            quantized_inputs[key] = dynamic_quantize(value, bit_width=8)
        
        # Forward pass
        output = self.model(quantized_inputs, state)
        
        return output
    
    def compute_loss(self, output, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss with quantization-aware components."""
        losses = {}
        
        # Predictive loss
        if 'z_pred' in target:
            pred_loss = F.mse_loss(output.z_pred, target['z_pred'])
            losses['predictive'] = pred_loss * self.config.predictive_weight
        
        # Medium horizon loss
        if 'medium_horizon' in target:
            mh_loss = F.mse_loss(output.z_pred, target['medium_horizon'])
            losses['medium_horizon'] = mh_loss * self.config.medium_horizon_weight
        
        # Policy outcome loss
        if 'policy_outcome' in target:
            po_loss = F.mse_loss(output.z_pred, target['policy_outcome'])
            losses['policy_outcome'] = po_loss * self.config.policy_outcome_weight
        
        # Belief consistency loss
        if 'belief_consistency' in target:
            bc_loss = F.mse_loss(output.z_pred, target['belief_consistency'])
            losses['belief_consistency'] = bc_loss * self.config.belief_consistency_weight
        
        # Ternary regularization
        ternary_reg = self.compute_ternary_regularization()
        losses['ternary_reg'] = ternary_reg
        
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss, losses
    
    def train_step(self, input_tensors: Dict[str, torch.Tensor],
                   target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform one training step with quantization awareness."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with quantization simulation
        output = self.forward_with_quantization(input_tensors)
        
        # Compute loss
        total_loss, losses = self.compute_loss(output, target)
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update state
        self.state.step += 1
        self.state.ternary_sparsity = self._compute_ternary_sparsity()
        
        return total_loss, losses
    
    def _compute_ternary_sparsity(self) -> float:
        """Compute ternary sparsity across model."""
        total_weights = 0
        ternary_weights = 0
        
        for module in self.model.modules():
            if isinstance(module, TernaryLinear):
                weights = module.weight
                total_weights += weights.numel()
                ternary_weights += (torch.abs(weights) > 0.1).sum().item()
        
        if total_weights == 0:
            return 0.0
        
        return ternary_weights / total_weights
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_state': self.state,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.state = checkpoint['training_state']


class BitNetQuantizationSimulator:
    """Simulates quantization for training-time analysis.
    
    Helps understand the impact of low-bit quantization on model behavior.
    """
    
    def __init__(self, model: BitNetACMoEGAModel):
        self.model = model
        self.quantization_errors = []
    
    def simulate_quantization(self, x: torch.Tensor, bit_width: int = 8) -> torch.Tensor:
        """Simulate quantization for analysis."""
        # Quantize
        max_val = torch.max(torch.abs(x))
        num_levels = 2 ** (bit_width - 1) - 1
        scale = max_val / num_levels
        
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, -num_levels, num_levels)
        
        x_dq = x_q * scale
        
        # Track quantization error
        error = F.mse_loss(x, x_dq).item()
        self.quantization_errors.append(error)
        
        return x_dq
    
    def analyze_quantization_impact(self) -> Dict[str, float]:
        """Analyze impact of quantization on model behavior."""
        if not self.quantization_errors:
            return {'avg_error': 0.0, 'max_error': 0.0}
        
        errors = self.quantization_errors
        return {
            'avg_error': sum(errors) / len(errors),
            'max_error': max(errors),
            'min_error': min(errors),
        }
