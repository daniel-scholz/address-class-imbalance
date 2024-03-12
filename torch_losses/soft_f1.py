import torch


class SoftF1LossMinorityClass(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.soft_f1_loss_fn = SoftF1LossWithLogits()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        # flip targets such that minority class is 1
        target = 1 - target
        pred = -pred  # flip logits such that minority class is < 0

        return self.soft_f1_loss_fn(pred, target)


class SoftF1LossWithLogits(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        tp = (target * pred).sum()
        # tn = ((1 - targets) * (1 - preds)).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()

        p = tp / (tp + fp + 1e-16)
        r = tp / (tp + fn + 1e-16)

        soft_f1 = 2 * p * r / (p + r + 1e-16)

        soft_f1 = soft_f1.mean()

        return 1 - soft_f1


class SoftF1LossMulti(torch.nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bin_loss_fn = SoftF1LossWithLogits()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        pred = torch.softmax(pred, dim=1)

        for i in range(self.num_classes):
            loss += self.bin_loss_fn(target[:, i], pred[:, i])

        loss /= self.num_classes

        return loss


if __name__ == "__main__":
    # test binary version
    loss_fn = SoftF1LossWithLogits()
    preds = torch.tensor([0.1, 0.2, 0.3, 0.4])
    labels = torch.tensor([0, 1, 0, 1])
    loss = loss_fn(preds, labels)
    print(loss)

    # test multi-class version
    num_classes = 4
    loss_fn = SoftF1LossMulti(num_classes)
    preds = torch.tensor([[-1.0, -0.5, 0.5, 1.0], [1.0, 0.5, -0.5, -1.0]])

    labels = torch.tensor([[0, 0, 0, 1], [0, 1, 0, 0]])

    loss = loss_fn(preds, labels)
    print(loss)
