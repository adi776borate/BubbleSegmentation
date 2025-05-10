import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, smooth=1e-6):
        super().__init__()
        self.delta = delta
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: [B, 2, H, W], targets: [B, 1, H, W] or [B, H, W]
        probs = F.softmax(preds, dim=1)[:, 1, :, :]  # take class 1 (foreground)
        if targets.ndim == 4:
            targets = targets[:, 0, :, :]
        targets = targets.float()

        TP = (probs * targets).sum(dim=(1, 2))
        FN = (targets * (1 - probs)).sum(dim=(1, 2))
        FP = ((1 - targets) * probs).sum(dim=(1, 2))

        tversky = (TP + self.smooth) / (TP + self.delta * FN + (1 - self.delta) * FP + self.smooth)
        return 1 - tversky.mean()



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')  # required for 2-class logits

    def forward(self, preds, targets):
        # preds: [B, 2, H, W], targets: [B, H, W] or [B, 1, H, W]
        if targets.ndim == 4:
            targets = targets[:, 0, :, :]
        targets = targets.long()

        ce_loss = self.ce(preds, targets)  # [B, H, W]
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal



class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, tversky_weight=0.5, focal_weight=0.5, delta=0.7):
        super().__init__()
        self.tversky = AsymmetricTverskyLoss(delta=delta)
        self.focal = FocalLoss(gamma=2.0, reduction='none')
        self.w_t = tversky_weight
        self.w_f = focal_weight

    def forward(self, preds, targets):
        tversky_loss = self.tversky(preds, targets)               # scalar
        focal_loss = self.focal(preds, targets).mean()            # scalar
        return self.w_t * tversky_loss + self.w_f * focal_loss