"""
Calibration for AC-MoE-GA Sidecar.

Adjusts the model's abstention threshold based on cache outcomes
to improve decision quality in deployment.
"""

from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class CalibrationStats:
    """Calibration statistics."""
    total_decisions: int = 0
    correct_decisions: int = 0
    abstentions: int = 0
    correct_abstentions: int = 0
    wrong_overrides: int = 0
    missed_opportunities: int = 0


class OutcomeCalibrator:
    """
    Calibrates the sidecar's abstention threshold based on cache outcomes.
    
    The key insight: if the model is making decisions with low confidence
    but the cache outcomes show it's wrong, we should increase abstention.
    """

    def __init__(
        self,
        initial_threshold: float = 0.20,
        min_threshold: float = 0.10,
        max_threshold: float = 0.50,
        learning_rate: float = 0.01,
        window_size: int = 100,
    ):
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.learning_rate = learning_rate
        self.window_size = window_size

        self.stats = CalibrationStats()
        self.outcome_window: List[bool] = []  # True = correct, False = wrong

    def record_decision(
        self,
        confidence: float,
        abstain: bool,
        cache_hit: bool,
        page_state_cold: float,
        page_state_hot: float,
    ):
        """
        Record a decision and its outcome.
        
        Args:
            confidence: Model's confidence in the decision
            abstain: Whether the model abstained
            cache_hit: Whether the cache hit (True) or miss (False)
            page_state_cold: Model's prediction of coldness
            page_state_hot: Model's prediction of hotness
        """
        self.stats.total_decisions += 1

        # Determine if the decision was correct
        # If cache hit: page was hot, so model should have predicted hot
        # If cache miss: page was cold, so model should have predicted cold
        if cache_hit:
            # Page was hot
            correct = page_state_hot > page_state_cold
        else:
            # Page was cold
            correct = page_state_cold > page_state_hot

        if abstain:
            self.stats.abstentions += 1
            if correct:
                self.stats.correct_abstentions += 1
        else:
            if correct:
                self.stats.correct_decisions += 1
            else:
                self.stats.wrong_overrides += 1

        # Track in window
        self.outcome_window.append(correct)
        if len(self.outcome_window) > self.window_size:
            self.outcome_window.pop(0)

        # Adjust threshold based on outcome
        self._adjust_threshold()

    def _adjust_threshold(self):
        """Adjust the abstention threshold based on recent outcomes."""
        if len(self.outcome_window) < 10:
            return

        # Calculate recent accuracy
        recent_accuracy = sum(self.outcome_window) / len(self.outcome_window)

        # If accuracy is low, increase abstention threshold
        # If accuracy is high, decrease abstention threshold
        if recent_accuracy < 0.5:
            # Low accuracy - increase abstention
            self.current_threshold = min(
                self.max_threshold,
                self.current_threshold + self.learning_rate
            )
        elif recent_accuracy > 0.7:
            # High accuracy - decrease abstention
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold - self.learning_rate * 0.5
            )

    def should_abstain(self, confidence: float) -> bool:
        """Determine if the model should abstain based on confidence."""
        return confidence < self.current_threshold

    def get_stats(self) -> Dict[str, float]:
        """Get calibration statistics."""
        total = self.stats.total_decisions
        return {
            "total_decisions": total,
            "accuracy": self.stats.correct_decisions / max(1, total - self.stats.abstentions),
            "abstention_rate": self.stats.abstentions / max(1, total),
            "correct_abstention_rate": self.stats.correct_abstentions / max(1, self.stats.abstentions),
            "wrong_override_rate": self.stats.wrong_overrides / max(1, total - self.stats.abstentions),
            "current_threshold": self.current_threshold,
        }


class MarginCalibrator:
    """
    Calibrates the sidecar's action margin threshold based on cache outcomes.
    
    The key insight: if the model is making decisions with low margins
    but the cache outcomes show it's wrong, we should increase the margin threshold.
    """

    def __init__(
        self,
        initial_threshold: float = 0.05,
        min_threshold: float = 0.01,
        max_threshold: float = 0.20,
        learning_rate: float = 0.005,
        window_size: int = 100,
    ):
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.learning_rate = learning_rate
        self.window_size = window_size

        self.outcome_window: List[bool] = []  # True = correct, False = wrong

    def record_decision(self, margin: float, cache_hit: bool, correct: bool):
        """Record a decision and its outcome."""
        self.outcome_window.append(correct)
        if len(self.outcome_window) > self.window_size:
            self.outcome_window.pop(0)

        # Adjust threshold based on outcome
        self._adjust_threshold()

    def _adjust_threshold(self):
        """Adjust the margin threshold based on recent outcomes."""
        if len(self.outcome_window) < 10:
            return

        # Calculate recent accuracy for high-margin decisions
        high_margin_correct = sum(
            c for m, c in zip(self.outcome_window[-20:], self.outcome_window[-20:])
            if m > self.current_threshold
        )
        high_margin_total = min(20, len(self.outcome_window))

        if high_margin_total > 0:
            high_margin_accuracy = high_margin_correct / high_margin_total
        else:
            high_margin_accuracy = 0.5

        # If high-margin decisions are often wrong, increase threshold
        if high_margin_accuracy < 0.6:
            self.current_threshold = min(
                self.max_threshold,
                self.current_threshold + self.learning_rate
            )
        elif high_margin_accuracy > 0.8:
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold - self.learning_rate * 0.5
            )

    def should_override(self, margin: float) -> bool:
        """Determine if the model should override based on margin."""
        return margin > self.current_threshold

    def get_stats(self) -> Dict[str, float]:
        """Get calibration statistics."""
        return {
            "current_threshold": self.current_threshold,
        }
