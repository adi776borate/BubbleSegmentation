import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: [B, 2, H, W], targets: [B, 1, H, W] or [B, H, W]
        probs = F.softmax(preds, dim=1)[:, 1, :, :]  # Use foreground prob
        if targets.ndim == 4:
            targets = targets[:, 0, :, :]
        targets = targets.float()

        preds_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)
        intersection = (preds_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()