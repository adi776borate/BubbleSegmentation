def plot_metrics_vs_pulses(metrics_csv_path, save_dir, experiment_name):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    metrics_df = pd.read_csv(metrics_csv_path)

    # Rename column if needed
    if 'Pulse' in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={'Pulse': 'pulses'})

    metrics_to_plot = {
        "GlobalAccuracy": "Predictive Accuracy (%)",
        "Dice (F1 Score)": "Dice Similarity Coefficient (%)",
        "MaxHausdorff": "Max Hausdorff Distance (mm)",
        "MeanHausdorff": "Mean Hausdorff Distance (mm)"
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (metric, ylabel) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]

        if metric not in metrics_df.columns:
            print(f"Warning: Metric '{metric}' not found in DataFrame columns.")
            continue

        grouped = metrics_df.groupby('pulses')[metric].agg(['mean', 'std']).reset_index().sort_values(by='pulses')

        ax.errorbar(grouped['pulses'], grouped['mean'], yerr=grouped['std'], fmt='o-', ecolor='gray', capsize=2, color='dodgerblue', label='CNN')
        ax.set_xlabel('Number of Pulses', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.suptitle(f"{experiment_name} Metrics vs. Number of Pulses", fontsize=16, y=1.02)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{experiment_name}_metrics_vs_pulses.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"--> Metrics plot saved to {plot_path}")


def plot_ablation_area_comparison(
    gt_folder,
    pred_folder,
    save_path,
    experiment_name,
    pixel_area_mm2=0.0025,
    filename_pattern=r't3Label(\d+)_(\d+)_(\d+)'
):
    import os
    import re
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from glob import glob

    def compute_ablation_areas(folder, label):
        data = []
        mask_files = glob(os.path.join(folder, "*.png"))
        for mask_file in mask_files:
            filename = os.path.basename(mask_file)
            match = re.match(filename_pattern, filename)
            if not match:
                continue
            pulses = int(match.group(1))
            experiment_id = match.group(2)
            dataset_idx = int(match.group(3))

            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARNING] Failed to read {label} mask: {mask_file}")
                continue
            _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            ablation_area = np.sum(binary_mask) * pixel_area_mm2

            data.append({
                'pulses': pulses,
                'experiment_id': experiment_id,
                'dataset_idx': dataset_idx,
                'ablation_area': ablation_area
            })
        return pd.DataFrame(data)

    # Compute from GT and Prediction masks
    gt_df = compute_ablation_areas(gt_folder, "Ground Truth")
    pred_df = compute_ablation_areas(pred_folder, "Prediction")

    if gt_df.empty or pred_df.empty:
        raise ValueError("Either ground truth or prediction ablation area data is empty. Check input folders or filenames.")

    gt_grouped = gt_df.groupby('pulses')['ablation_area'].agg(['mean', 'std']).reset_index()
    pred_grouped = pred_df.groupby('pulses')['ablation_area'].agg(['mean', 'std']).reset_index()

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.errorbar(gt_grouped['pulses'], gt_grouped['mean'], yerr=gt_grouped['std'],
                 fmt='*-', capsize=3, label='Ground Truth', color='orange')
    plt.errorbar(pred_grouped['pulses'], pred_grouped['mean'], yerr=pred_grouped['std'],
                 fmt='o-', capsize=3, label='Prediction', color='blue')

    plt.xlabel("Number of Pulses", fontsize=14)
    plt.ylabel("Mean Ablation Area (mm²)", fontsize=14)
    plt.title("Ground Truth vs Predicted Ablation Area vs Pulses", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    final_path = os.path.join(save_path, f"{experiment_name}_ablation_area_vs_pulses.png")
    plt.savefig(final_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"--> Ablation area comparison plot saved to {final_path}")


def get_binary_segmentation_predictions(model, images):
    import torch
    """
    Unified prediction function for binary segmentation.
    Returns class index masks with shape [B, H, W], values in {0, 1}.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(images)

        # Torchvision models
        if isinstance(outputs, dict) and 'out' in outputs:
            logits = outputs['out']
        # SMP / custom models
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        # Mask R-CNN
        elif isinstance(outputs, list) and isinstance(outputs[0], dict) and 'masks' in outputs[0]:
            batch_preds = []
            for instance in outputs:
                if len(instance["masks"]) == 0:
                    mask = torch.zeros_like(images[0, 0])  # [H, W]
                else:
                    mask = instance["masks"].squeeze(1).sum(dim=0)
                    mask = (mask > 0.5).float()
                batch_preds.append(mask)
            return torch.stack(batch_preds).long()  # [B, H, W]
        else:
            raise ValueError("Unsupported model output format.")

        # Handle binary logits
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze(1)  # [B, H, W]
        # Handle 2-class logits
        elif logits.shape[1] == 2:
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")

    return preds  # [B, H, W], dtype: long, values: 0 or 1