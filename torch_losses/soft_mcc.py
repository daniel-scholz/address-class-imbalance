import torch
import torch.nn as nn


# binary versions of the loss
class SoftMCCLoss(nn.Module):

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        tp = torch.sum(preds * labels)
        tn = torch.sum((1 - preds) * (1 - labels))
        fp = torch.sum(preds * (1 - labels))
        fn = torch.sum((1 - preds) * labels)

        numerator = tp * tn - fp * fn
        denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
        soft_mcc = numerator / denom

        loss = 1 - soft_mcc
        return loss


class SoftMCCWithLogitsLoss(SoftMCCLoss):

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        preds_sigmoid = torch.sigmoid(preds)
        return super().forward(preds_sigmoid, labels)


# multi-class versions of the loss
class SoftMCCLossMulti(nn.Module):
    """With logits."""

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        # create soft confusion matrix
        preds = torch.softmax(preds, dim=1)

        # total number of correct predictions, softened by the probability of each class
        c = torch.sum(preds * labels)

        # total number of samples
        s = preds.size(0)

        # number of times each class occured in the labels
        t_k = torch.sum(labels, dim=0)

        # number of times each class was predicted
        p_k = torch.sum(preds, dim=0)

        numerator = c * s - (t_k * p_k).sum()
        denom = (
            torch.sqrt(s**2 - p_k.square().sum())
            * torch.sqrt(s**2 - t_k.square().sum())
            + 1e-8
        )

        soft_mcc = numerator / denom
        return 1 - soft_mcc


if __name__ == "__main__":
    # test binary version
    loss_fn = SoftMCCLoss()
    preds = torch.tensor([0.1, 0.2, 0.3, 0.4])
    labels = torch.tensor([0, 1, 0, 1])
    loss = loss_fn(preds, labels)
    print(loss)

    # test multi-class version
    loss_fn = SoftMCCLossMulti()
    preds = torch.tensor([[-1.0, -0.5, 0.5, 1.0], [1.0, 0.5, -0.5, -1.0]])

    labels = torch.tensor([[0, 0, 0, 1], [0, 1, 0, 0]])
    print(preds.size(), labels.size())

    loss = loss_fn(preds, labels)
    print(loss)
