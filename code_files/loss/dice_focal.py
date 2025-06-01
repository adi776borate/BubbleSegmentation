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



class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=2.0)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds, targets):
        dice = self.dice(preds, targets)
        focal = self.focal(preds, targets)
        return self.dice_weight * dice + self.focal_weight * focal

# === Temporary Debug Loss Function ===
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, preds, targets):
#         preds = F.softmax(preds, dim=1)[:, 1, :, :]  # class 1
#         if targets.ndim == 4:
#             targets = targets[:, 0, :, :]
#         targets = (targets == 1).float()
        
#         intersection = (preds * targets).sum()
#         dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
#         return 1 - dice

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super().__init__()
#         self.gamma = gamma
#         self.ce = nn.CrossEntropyLoss()

#     def forward(self, preds, targets):
#         if targets.ndim == 4:  # [B, 1, H, W]
#             targets = targets[:, 0, :, :]
#         targets = targets.long()
        
#         ce_loss = self.ce(preds, targets)
#         pt = torch.exp(-ce_loss)
#         focal = (1 - pt) ** self.gamma * ce_loss
#         return focal

# class DiceFocalLoss(nn.Module):
#     def __init__(self, dice_weight=0.5, focal_weight=0.5):
#         super().__init__()
#         self.dice = DiceLoss()
#         self.focal = FocalLoss()
#         self.dice_weight = dice_weight
#         self.focal_weight = focal_weight

#     def forward(self, preds, targets):
#         return self.dice_weight * self.dice(preds, targets) + self.focal_weight * self.focal(preds, targets)
