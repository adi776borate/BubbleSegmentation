import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.ndimage import binary_erosion, distance_transform_edt
import warnings
from config import Config

# --- Extract binary boundary map ---
def get_boundary(mask):
    eroded = binary_erosion(mask, border_value=0)
    boundary = mask ^ eroded
    return boundary

# --- Boundary F1 score (MATLAB-style) ---
def compute_bf_score_single(pred, target, tolerance=2):
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    pred_boundary = get_boundary(pred)
    target_boundary = get_boundary(target)

    dt_pred = distance_transform_edt(~pred_boundary)
    dt_target = distance_transform_edt(~target_boundary)

    match_pred = target_boundary & (dt_pred <= tolerance)
    match_target = pred_boundary & (dt_target <= tolerance)

    precision = match_target.sum() / (pred_boundary.sum() + 1e-8)
    recall = match_pred.sum() / (target_boundary.sum() + 1e-8)

    if precision + recall == 0:
        return 0.0

    bf_score = 2 * precision * recall / (precision + recall)
    return bf_score

def compute_mean_bf_score(preds, targets, tolerance=2):
    scores = []
    for i in range(preds.shape[0]):
        pred = preds[i, 0].astype(np.bool_)
        target = targets[i, 0].astype(np.bool_)
        score = compute_bf_score_single(pred, target, tolerance)
        scores.append(score)
    return float(np.mean(scores)) if scores else -1.0

# --- Hausdorff Distance ---
def compute_hausdorff(preds, targets):
    mean_hd, max_hd = [], []
    skipped = 0

    for i in range(preds.shape[0]):
        # FIXED: remove [0] indexing
        p = preds[i].astype(np.bool_)
        t = targets[i].astype(np.bool_)

        pred_boundary = get_boundary(p)
        target_boundary = get_boundary(t)

        p_coords = np.argwhere(pred_boundary)
        t_coords = np.argwhere(target_boundary)

        if p_coords.size == 0 or t_coords.size == 0:
            skipped += 1
            continue

        try:
            hd1 = directed_hausdorff(p_coords, t_coords)[0]
            hd2 = directed_hausdorff(t_coords, p_coords)[0]
            max_hd.append(max(hd1, hd2))

            dist1 = cdist(p_coords, t_coords).min(axis=1)
            dist2 = cdist(t_coords, p_coords).min(axis=1)
            mean_hd.append((dist1.mean() + dist2.mean()) / 2)
        except Exception as e:
            raise RuntimeError(f"Hausdorff calculation failed for sample {i}: {e}")

    if not mean_hd or not max_hd:
        raise ValueError(
            f"Hausdorff computation failed for all {preds.shape[0]} samples. "
            f"Possible causes: all-empty predictions, incorrect shape, or broken mask."
        )

    return float(np.mean(mean_hd)), float(np.mean(max_hd))

# --- Per-class metrics ---
def compute_class_metrics(y_true, y_pred, class_val, epsilon=1e-7):
    cls_pred = (y_pred == class_val)
    cls_true = (y_true == class_val)

    TP = np.sum(cls_pred & cls_true)
    TN = np.sum(~cls_pred & ~cls_true)
    FP = np.sum(cls_pred & ~cls_true)
    FN = np.sum(~cls_pred & cls_true)

    accuracy = TP / (TP + FN + epsilon)
    iou = TP / (TP + FP + FN + epsilon)
    
    dice = (2 * TP) / (2 * TP + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = (2 * precision * recall) / (precision + recall + epsilon)

    return accuracy, iou, dice, precision, recall, f1_score, TP, FP, FN

# --- Main Metric Function ---
def calculate_all_metrics(predictions, targets, threshold=0.5):
    # Ensure shape is [B, H, W]
    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]

    assert predictions.shape == targets.shape and predictions.ndim == 3, "Expecting [B, H, W] for both predictions and targets"


    probs = torch.sigmoid(predictions)
    preds_bin = (probs > threshold).int()
    
    preds_np = preds_bin.cpu().numpy()
    targets_np = targets.cpu().numpy()

    y_true_flat = targets_np.flatten()
    y_pred_flat = preds_np.flatten()

    global_acc = np.sum(y_pred_flat == y_true_flat) / len(y_true_flat)

    class_ids = [0, 1]
    class_accuracies, class_ious, class_dices = [], [], []
    class_precisions, class_recalls, class_f1s = [], [], []
    weighted_ious = []
    TP_sum, FP_sum, FN_sum = 0, 0, 0

    for cls in class_ids:
        acc, iou, dice, prec, rec, f1, TP, FP, FN = compute_class_metrics(y_true_flat, y_pred_flat, cls)
        class_accuracies.append(acc)
        class_ious.append(iou)
        class_dices.append(dice)
        class_precisions.append(prec)
        class_recalls.append(rec)
        class_f1s.append(f1)
        class_px_count = np.sum(y_true_flat == cls)
        weighted_ious.append(iou * class_px_count)
        TP_sum += TP
        FP_sum += FP
        FN_sum += FN

    mean_acc = np.mean(class_accuracies)
    mean_iou = np.mean(class_ious)
    weighted_iou = np.sum(weighted_ious) / (len(y_true_flat) + 1e-7)
    mean_dice = np.mean(class_dices)
    mean_precision = np.mean(class_precisions)
    mean_recall = np.mean(class_recalls)

    mean_bf_score = compute_mean_bf_score(preds_np, targets_np)
    mean_hd, max_hd = compute_hausdorff(preds_np, targets_np)

    try:
        probs_flat = probs.detach().cpu().numpy().flatten()
        if len(np.unique(y_true_flat)) > 1:
            auroc = roc_auc_score(y_true_flat, probs_flat)
        else:
            auroc = 0.5
    except Exception as e:
        warnings.warn(f"AUROC computation failed: {e}")
        auroc = 0.5

    return {
        "GlobalAccuracy": global_acc,
        "MeanAccuracy": mean_acc,
        "MeanIoU": mean_iou,
        "WeightedIoU": weighted_iou,
        "MeanBFScore": mean_bf_score,
        "Dice (F1 Score)": mean_dice,
        "AUROC": auroc,
        "Precision": mean_precision,
        "Recall": mean_recall,
        "MeanHausdorff": mean_hd,
        "MaxHausdorff": max_hd,
    }