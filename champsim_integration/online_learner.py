"""
Online learning for AC-MoE-GA Sidecar.

Trains the model based on cache outcomes (hit/miss) to improve
page state predictions in deployment.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass

from ac_moe_ga_sidecar.types import PageState
from ac_moe_ga_sidecar.training import TrainingBatch


@dataclass
class DecisionRecord:
    """Record of a sidecar decision for online learning."""
    features: Dict[str, torch.Tensor]
    page_state: PageState
    cache_hit: Optional[bool]  # None if not yet observed
    timestamp: int


class OnlineLearner:
    """
    Online learner that trains the sidecar based on cache outcomes.
    
    The key insight: cache hit/miss tells us whether a page was hot or cold.
    We use this to train the model's page state prediction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        max_history: int = 1000,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_history = max_history

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.decision_history: List[DecisionRecord] = []

    def record_decision(
        self,
        features: Dict[str, torch.Tensor],
        page_state: PageState,
        timestamp: int,
    ):
        """Record a decision for later training."""
        record = DecisionRecord(
            features=features,
            page_state=page_state,
            cache_hit=None,
            timestamp=timestamp,
        )
        self.decision_history.append(record)

        # Trim history if needed
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]

    def record_outcome(self, timestamp: int, cache_hit: bool):
        """Record the cache outcome for a decision."""
        # Find the most recent decision without an outcome
        for record in reversed(self.decision_history):
            if record.timestamp < timestamp and record.cache_hit is None:
                record.cache_hit = cache_hit
                break

    def train_step(self) -> Dict[str, float]:
        """
        Train the model on recorded decisions with outcomes.
        
        Uses cache hit/miss as supervision for page state prediction.
        """
        if len(self.decision_history) < self.batch_size:
            return {"loss": 0.0, "count": 0}

        # Find decisions with outcomes
        decisions_with_outcomes = [
            d for d in self.decision_history
            if d.cache_hit is not None
        ]

        if len(decisions_with_outcomes) < self.batch_size:
            return {"loss": 0.0, "count": len(decisions_with_outcomes)}

        # Sample a batch
        batch = decisions_with_outcomes[-self.batch_size:]

        # Extract features and labels
        feature_dict = batch[0].features  # Use first decision's feature keys
        batch_features = {
            key: torch.stack([d.features[key] for d in batch])
            for key in feature_dict.keys()
        }
        labels = torch.tensor([d.cache_hit for d in batch], dtype=torch.float32)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_features)

        # Compute loss: encourage hot pages to be predicted as hot
        # PageState has: cold, recently_reused, burst_hot, likely_write_hot_soon, reclaimable, fault_prone, cow_sensitive, hugepage_friendly, unstable
        # Hot pages: high burst_hot or recently_reused
        # Cold pages: high cold or reclaimable

        # Predicted hotness = burst_hot + recently_reused
        predicted_hotness = (
            outputs["page_state"]["burst_hot"] +
            outputs["page_state"]["recently_reused"]
        )
        predicted_coldness = (
            outputs["page_state"]["cold"] +
            outputs["page_state"]["reclaimable"]
        )

        # Loss: encourage hot pages to have high predicted_hotness
        # and cold pages to have high predicted_coldness
        hot_loss = F.mse_loss(
            predicted_hotness[labels == 1],
            torch.ones_like(predicted_hotness[labels == 1]) * 0.8
        ) if (labels == 1).any() else 0.0

        cold_loss = F.mse_loss(
            predicted_coldness[labels == 0],
            torch.ones_like(predicted_coldness[labels == 0]) * 0.8
        ) if (labels == 0).any() else 0.0

        total_loss = hot_loss + cold_loss

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "hot_loss": hot_loss.item() if isinstance(hot_loss, torch.Tensor) else hot_loss,
            "cold_loss": cold_loss.item() if isinstance(cold_loss, torch.Tensor) else cold_loss,
            "count": len(batch),
        }

    def get_stats(self) -> Dict[str, float]:
        """Get online learning statistics."""
        decisions_with_outcomes = [
            d for d in self.decision_history
            if d.cache_hit is not None
        ]
        return {
            "total_decisions": len(self.decision_history),
            "decisions_with_outcomes": len(decisions_with_outcomes),
            "hot_pages": sum(1 for d in decisions_with_outcomes if d.cache_hit),
            "cold_pages": sum(1 for d in decisions_with_outcomes if not d.cache_hit),
        }
