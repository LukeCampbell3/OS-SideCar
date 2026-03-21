"""
Training module for AC-MoE-GA Sidecar.

Implements the multi-objective training pipeline including:
- Predictive objectives
- Medium-horizon objectives
- Policy-outcome objectives
- Belief-consistency objectives
- Routing objectives
- Uncertainty calibration objectives
- Distillation objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .config import SidecarConfig, TrainingConfig
from .model import ACMoEGAModel, ModelState


@dataclass
class TrainingBatch:
    """A training batch of event sequences."""
    # Input features (batch, seq_len, feature_dim)
    inputs: Dict[str, torch.Tensor]
    
    # Short-horizon labels
    touch_soon: torch.Tensor  # (batch, seq_len)
    write_soon: torch.Tensor
    fault_soon: torch.Tensor
    cow_soon: torch.Tensor
    reclaim_safe: torch.Tensor
    
    # Medium-horizon labels
    sustained_hotness: torch.Tensor
    region_drift: torch.Tensor
    fault_storm_onset: torch.Tensor
    working_set_shift: torch.Tensor
    
    # Policy outcome labels
    preserve_vs_reclaim: torch.Tensor  # 1 if preserve was better
    batching_helped: torch.Tensor
    locality_helped: torch.Tensor
    
    # Uncertainty labels
    was_wrong_high_conf: torch.Tensor  # For calibration
    was_ood: torch.Tensor
    
    # Sequence mask
    mask: torch.Tensor  # (batch, seq_len)


class SidecarLoss(nn.Module):
    """
    Multi-objective loss function for AC-MoE-GA Sidecar.
    
    Combines multiple loss terms with configurable weights.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: TrainingBatch,
        routing_weights: torch.Tensor,
        expert_gains: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and component losses.
        
        Returns:
            total_loss: Scalar loss for backprop
            loss_dict: Dictionary of individual loss components
        """
        mask = batch.mask
        
        losses = {}
        
        # === Predictive objectives ===
        pred_loss = self._predictive_loss(outputs, batch, mask)
        losses['predictive'] = pred_loss
        
        # === Medium-horizon objectives ===
        horizon_loss = self._medium_horizon_loss(outputs, batch, mask)
        losses['medium_horizon'] = horizon_loss
        
        # === Policy-outcome objectives ===
        policy_loss = self._policy_outcome_loss(outputs, batch, mask)
        losses['policy_outcome'] = policy_loss
        
        # === Belief-consistency objectives ===
        belief_loss = self._belief_consistency_loss(outputs, mask)
        losses['belief_consistency'] = belief_loss
        
        # === Routing objectives ===
        routing_loss = self._routing_loss(routing_weights, expert_gains, mask)
        losses['routing'] = routing_loss
        
        # === Uncertainty calibration objectives ===
        uncertainty_loss = self._uncertainty_loss(outputs, batch, mask)
        losses['uncertainty'] = uncertainty_loss
        
        # === Distillation objectives ===
        distill_loss = self._distillation_loss(outputs, expert_gains, mask)
        losses['distillation'] = distill_loss
        
        # Combine with weights
        total_loss = (
            self.config.predictive_weight * pred_loss +
            self.config.medium_horizon_weight * horizon_loss +
            self.config.policy_outcome_weight * policy_loss +
            self.config.belief_consistency_weight * belief_loss +
            self.config.routing_sparsity_weight * routing_loss +
            self.config.uncertainty_calibration_weight * uncertainty_loss +
            self.config.distillation_weight * distill_loss
        )
        
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _predictive_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: TrainingBatch,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predictive loss for short-horizon targets."""
        page_outputs = outputs['page_state']
        
        # Touch soon
        touch_loss = self.bce(
            page_outputs['touch_soon'],
            batch.touch_soon.float()
        )
        
        # Write soon
        write_loss = self.bce(
            page_outputs['write_soon'],
            batch.write_soon.float()
        )
        
        # Fault soon
        fault_loss = self.bce(
            page_outputs['fault_soon'],
            batch.fault_soon.float()
        )
        
        # COW risk
        cow_loss = self.bce(
            page_outputs['cow_risk'],
            batch.cow_soon.float()
        )
        
        # Reclaim safe
        reclaim_loss = self.bce(
            page_outputs['reclaim_safe'],
            batch.reclaim_safe.float()
        )
        
        # Combine and mask
        total = touch_loss + write_loss + fault_loss + cow_loss + reclaim_loss
        return (total * mask).sum() / mask.sum()
    
    def _medium_horizon_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: TrainingBatch,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute medium-horizon prediction loss."""
        region_outputs = outputs['region_state']
        
        # Sustained hotness (from page hotness_up)
        hotness_loss = self.bce(
            outputs['page_state']['hotness_up'],
            batch.sustained_hotness.float()
        )
        
        # Region drift (from volatile)
        drift_loss = self.bce(
            region_outputs['volatile'],
            batch.region_drift.float()
        )
        
        # Fault storm onset
        fault_storm_loss = self.bce(
            outputs['page_state']['fault_soon'],
            batch.fault_storm_onset.float()
        )
        
        total = hotness_loss + drift_loss + fault_storm_loss
        return (total * mask).sum() / mask.sum()
    
    def _policy_outcome_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: TrainingBatch,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy-outcome alignment loss."""
        # Page policy should align with preserve_vs_reclaim outcome
        page_actions = outputs['page_actions']
        
        # Softmax to get probabilities
        probs = F.softmax(page_actions, dim=-1)
        
        # Preserve action should be high when preserve was better
        preserve_prob = probs[:, 0]  # PRESERVE action
        reclaim_prob = probs[:, 1]  # RECLAIM_CANDIDATE action
        
        # Loss: encourage preserve when preserve_vs_reclaim=1
        policy_loss = self.bce(
            preserve_prob - reclaim_prob,
            batch.preserve_vs_reclaim.float() * 2 - 1  # Map to [-1, 1]
        )
        
        return (policy_loss * mask).sum() / mask.sum()
    
    def _belief_consistency_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage smooth belief evolution."""
        # This would require access to belief states over time
        # For now, use uncertainty as proxy
        uncertainty = outputs['uncertainty']
        
        # Penalize high uncertainty variance (encourage stability)
        if uncertainty.dim() > 1 and uncertainty.shape[0] > 1:
            uncertainty_var = uncertainty.var(dim=0).mean()
        else:
            uncertainty_var = torch.tensor(0.0, device=uncertainty.device)
        
        return uncertainty_var
    
    def _routing_loss(
        self,
        routing_weights: torch.Tensor,
        expert_gains: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage sparse and effective expert routing."""
        # Sparsity: penalize using many experts
        sparsity_loss = (routing_weights > 0.1).float().sum(dim=-1).mean()
        
        # Effectiveness: penalize routing to low-gain experts
        if expert_gains.sum() > 0:
            gain_weighted = (routing_weights * expert_gains.unsqueeze(0)).sum(dim=-1)
            effectiveness_loss = -gain_weighted.mean()
        else:
            effectiveness_loss = torch.tensor(0.0, device=routing_weights.device)
        
        return sparsity_loss + 0.1 * effectiveness_loss
    
    def _uncertainty_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: TrainingBatch,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrate uncertainty estimates."""
        uncertainty = outputs['uncertainty']
        
        # Calibration: high uncertainty should correlate with being wrong
        calibration_uncertainty = uncertainty[:, 0]  # First component
        calibration_loss = self.bce(
            calibration_uncertainty,
            batch.was_wrong_high_conf.float()
        )
        
        # OOD: OOD uncertainty should be high for OOD samples
        ood_uncertainty = uncertainty[:, 3]
        ood_loss = self.bce(
            ood_uncertainty,
            batch.was_ood.float()
        )
        
        total = calibration_loss + ood_loss
        return (total * mask).sum() / mask.sum()
    
    def _distillation_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        expert_gains: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage dense path to learn from expert gains."""
        # This would require storing expert outputs separately
        # For now, return zero
        return torch.tensor(0.0, device=mask.device)


class Trainer:
    """
    Training orchestrator for AC-MoE-GA Sidecar.
    """
    
    def __init__(
        self,
        model: ACMoEGAModel,
        config: SidecarConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.loss_fn = SidecarLoss(config.training)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
        )
        
        # Training statistics
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.loss_history: List[Dict[str, float]] = []
        
    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """Execute a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = self._to_device(batch)
        
        # Forward pass
        state = None
        all_outputs = []
        all_routing = []
        
        seq_len = batch.mask.shape[1]
        for t in range(seq_len):
            # Get inputs for this timestep
            inputs_t = {k: v[:, t] if v.dim() > 2 else v for k, v in batch.inputs.items()}
            
            output = self.model(inputs_t, state)
            state = output.new_state
            
            all_outputs.append(output.head_outputs)
            all_routing.append(output.routing_weights)
        
        # Stack outputs
        stacked_outputs = self._stack_outputs(all_outputs)
        routing_weights = torch.stack(all_routing, dim=1)
        
        # Compute loss
        loss, loss_dict = self.loss_fn(
            stacked_outputs,
            batch,
            routing_weights.mean(dim=1),  # Average over sequence
            self.model.expert_router.expert_gains,
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        
        self.step += 1
        self.loss_history.append(loss_dict)
        
        return loss_dict
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._to_device(batch)
                
                state = None
                all_outputs = []
                all_routing = []
                
                seq_len = batch.mask.shape[1]
                for t in range(seq_len):
                    inputs_t = {k: v[:, t] if v.dim() > 2 else v for k, v in batch.inputs.items()}
                    output = self.model(inputs_t, state)
                    state = output.new_state
                    all_outputs.append(output.head_outputs)
                    all_routing.append(output.routing_weights)
                
                stacked_outputs = self._stack_outputs(all_outputs)
                routing_weights = torch.stack(all_routing, dim=1)
                
                _, loss_dict = self.loss_fn(
                    stacked_outputs,
                    batch,
                    routing_weights.mean(dim=1),
                    self.model.expert_router.expert_gains,
                )
                
                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0) + v
                num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
    
    def _to_device(self, batch: TrainingBatch) -> TrainingBatch:
        """Move batch to device."""
        return TrainingBatch(
            inputs={k: v.to(self.device) for k, v in batch.inputs.items()},
            touch_soon=batch.touch_soon.to(self.device),
            write_soon=batch.write_soon.to(self.device),
            fault_soon=batch.fault_soon.to(self.device),
            cow_soon=batch.cow_soon.to(self.device),
            reclaim_safe=batch.reclaim_safe.to(self.device),
            sustained_hotness=batch.sustained_hotness.to(self.device),
            region_drift=batch.region_drift.to(self.device),
            fault_storm_onset=batch.fault_storm_onset.to(self.device),
            working_set_shift=batch.working_set_shift.to(self.device),
            preserve_vs_reclaim=batch.preserve_vs_reclaim.to(self.device),
            batching_helped=batch.batching_helped.to(self.device),
            locality_helped=batch.locality_helped.to(self.device),
            was_wrong_high_conf=batch.was_wrong_high_conf.to(self.device),
            was_ood=batch.was_ood.to(self.device),
            mask=batch.mask.to(self.device),
        )
    
    def _stack_outputs(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack sequence of outputs."""
        stacked = {}
        for key in outputs[0].keys():
            if isinstance(outputs[0][key], dict):
                stacked[key] = {
                    k: torch.stack([o[key][k] for o in outputs], dim=1)
                    for k in outputs[0][key].keys()
                }
            else:
                stacked[key] = torch.stack([o[key] for o in outputs], dim=1)
        return stacked
