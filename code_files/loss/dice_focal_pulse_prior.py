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

class DiceFocalWithPulsePriorLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.6, prior_weight=0.35, alpha=0.15, beta=70):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=2.0)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.prior_weight = prior_weight
        self.alpha = alpha
        self.beta = beta

    def sigmoid_prior_from_pulse(self, pulse_values, shape):
        # pulse_values: (B,)
        # Output: (B, H, W)
        prior_probs = torch.sigmoid(self.alpha * (pulse_values.view(-1, 1, 1) - self.beta))  # (B, 1, 1)
        return prior_probs.expand(shape[0], shape[2], shape[3])  # (B, H, W)

    def forward(self, preds, targets, pulses):
        # preds: [B, 2, H, W], targets: [B, 1, H, W] or [B, H, W]
        dice = self.dice(preds, targets)
        focal = self.focal(preds, targets)

        # Predicted probability of foreground class (class 1)
        probs_fg = F.softmax(preds, dim=1)[:, 1, :, :]  # shape: (B, H, W)

        # Build pulse prior map
        prior_map = self.sigmoid_prior_from_pulse(pulses.to(preds.device), probs_fg.shape)

        # MSE between model confidence and prior map
        prior_penalty = F.mse_loss(probs_fg, prior_map)

        total = self.dice_weight * dice + self.focal_weight * focal + self.prior_weight * prior_penalty
        return total
