import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probas = torch.sigmoid(inputs)
        p_t = targets * probas + (1 - targets) * (1 - probas)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_term = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
