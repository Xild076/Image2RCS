import sys
import unittest
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import HybridRCSLoss, RelativeHuberLoss, reduce_per_sample_loss


class TestWeightedLoss(unittest.TestCase):
    def _assert_weighted_matches_repeated(self, criterion):
        predictions = torch.tensor([[0.12], [1.25], [0.45]], dtype=torch.float32)
        targets = torch.tensor([[0.20], [0.95], [0.40]], dtype=torch.float32)
        weights = torch.tensor([2, 3, 1], dtype=torch.float32)

        weighted_loss = reduce_per_sample_loss(criterion(predictions, targets), weights)

        repeat_indices = torch.repeat_interleave(torch.arange(predictions.shape[0]), weights.to(torch.long))
        repeated_predictions = predictions[repeat_indices]
        repeated_targets = targets[repeat_indices]
        repeated_loss = reduce_per_sample_loss(criterion(repeated_predictions, repeated_targets), None)

        self.assertAlmostEqual(weighted_loss.item(), repeated_loss.item(), places=6)

    def test_smoothl1_weighted_matches_repeated(self):
        criterion = nn.SmoothL1Loss(beta=0.25, reduction="none")
        self._assert_weighted_matches_repeated(criterion)

    def test_relative_huber_weighted_matches_repeated(self):
        criterion = RelativeHuberLoss(
            target_mode="none",
            relative_floor=0.05,
            beta=0.2,
            reduction="none",
        )
        self._assert_weighted_matches_repeated(criterion)

    def test_hybrid_weighted_matches_repeated(self):
        criterion = HybridRCSLoss(
            target_mode="none",
            alpha=0.7,
            smoothl1_beta=0.25,
            relative_floor=0.05,
            relative_beta=0.2,
            reduction="none",
        )
        self._assert_weighted_matches_repeated(criterion)


if __name__ == "__main__":
    unittest.main()
