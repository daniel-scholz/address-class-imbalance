from typing import Optional

import torch
import torch.nn as nn
from soft_mcc import SoftMCCLossMulti


class WeightedCombinedLosses(nn.Module):

    def __init__(
        self, losses: list[nn.Module], weights: Optional[list[float]] = None
    ) -> None:
        super().__init__()
        self.losses = losses
        # equal weights if not provided
        self.weights = (
            weights
            if weights is not None
            else [1 / len(self.losses)] * len(self.losses)
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=preds.device)
        for w, l in zip(self.weights, self.losses):
            loss += w * l(preds, targets)

        return loss


if __name__ == "__main__":
    losses = [
        nn.CrossEntropyLoss(),
        SoftMCCLossMulti(),
    ]

    weights = [1.0, 1.0]
    combined_loss = WeightedCombinedLosses(losses, weights)

    preds = torch.rand(100, 3)
    labels = torch.tensor([[0, 1, 0]] * 100, dtype=torch.float32)

    loss = combined_loss(preds, labels)

    print(loss)
