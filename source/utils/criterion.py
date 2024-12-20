import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑（Label Smoothing）是一种正则化技术，通常用于分类问题。
    它的目的是防止模型在训练时过于自信地预测标签，从而改善模型的泛化能力。
    """
    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = n_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
