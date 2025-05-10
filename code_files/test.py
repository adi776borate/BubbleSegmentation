import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tqdm import tqdm
from datetime import datetime
import warnings
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import re
from glob import glob
from torch.utils.data import DataLoader

from config import Config
from model import *
from loss import *
from metric import calculate_all_metrics
from dataloader import BubbleDataset
from train import get_model, get_loss_fn, load_checkpoint
from utils import plot_metrics_vs_pulses, plot_ablation_area_comparison, get_binary_segmentation_predictions

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

def recover_original(img, final_width=1024, final_height=256, cropped_width=750, cropped_height=554):
    img_resized = TF.resize(img, [cropped_height, cropped_width], interpolation=Image.NEAREST)
    pad_left = (final_width - cropped_width) // 2
    pad_right = final_width - cropped_width - pad_left
    pad_top = (final_height - cropped_height) // 2
    pad_bottom = final_height - cropped_height - pad_top
    return ImageOps.expand(img_resized, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

def extract_pulse(filename):
    base = os.path.basename(filename)
    pulse_match = re.search(r'US(\d+)', base)
    pulse = int(pulse_match.group(1)) if pulse_match else -1
    return pulse

def log_test_metrics_to_csv(log_path, config, metrics_dict):
    file_exists = os.path.isfile(log_path)
    header = [
        'Timestamp', 'Experiment_Name', 'Model_Name', 'Optimizer',
        'Loss_Function', 'Learning_Rate', 'Batch_Size'
    ]
    metric_keys = sorted([key for key in metrics_dict if pd.notna(metrics_dict[key])])
    header.extend(metric_keys)

    data_row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Experiment_Name': config.EXPERIMENT_NAME,
        'Model_Name': config.MODEL_NAME,
        'Optimizer': config.OPTIMIZER,
        'Loss_Function': config.LOSS_FN,
        'Learning_Rate': f"{config.LEARNING_RATE:.1E}",
        'Batch_Size': config.BATCH_SIZE
    }

    for key in metric_keys:
        value = metrics_dict[key]
        data_row[key] = f"{value:.6f}" if isinstance(value, (float, np.number)) else str(value)

    try:
        with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction='ignore')
            if not file_exists or os.path.getsize(log_path) == 0:
                writer.writeheader()
            writer.writerow(data_row)
    except Exception as e:
        print(f"[ERROR] Failed to write test metrics to CSV: {e}")

def compute_area(mask):
    return (mask == 1).sum().item()

# Parse pulse and dataset from filename
def extract_pulse_and_dataset(filename):
    base = os.path.basename(filename).replace(".jpg", "").replace(".png", "")
    parts = base.split('_')
    dataset = int(parts[-1]) if parts[-1].isdigit() else -1
    pulse_match = re.search(r'US(\d+)', base)
    pulse = int(pulse_match.group(1)) if pulse_match else -1
    return pulse, dataset


def main():
    import cv2
    config = Config()

    try:
        test_images = sorted(glob('../Data/US_Test_2023April7/*.jpg'))
        test_labels = sorted(glob('../Data/Label_Test_2023April7/*.png'))

        test_dataset = BubbleDataset(test_images, test_labels, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        model = get_model(config)
        criterion = get_loss_fn(config)
    except (AttributeError, ValueError, FileNotFoundError, ImportError) as e:
        print(f"Error during setup: {e}")
        return

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, "best.pth.tar")
    # checkpoint_path = "best_model_final.pth"
    if os.path.isfile(checkpoint_path):
        load_checkpoint(checkpoint_path, model, None, 0, config.DEVICE)
    else:
        print(f"ERROR: No checkpoint found at {checkpoint_path}.")
        return

    print("\n--- Starting Evaluation ---")
    all_samples = []
    individual_metrics = []

    # Create output folders
    vis_dir = os.path.join("test_results", config.EXPERIMENT_NAME, "visualizations")
    pred_dir = os.path.join("test_results", config.EXPERIMENT_NAME, "only_predictions")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    records = []
    all_mean_ious = []
    total_intersection = 0
    total_union = 0

    with torch.no_grad():
        for imgs, masks, orig_imgs, _, _ in tqdm(test_loader, desc="Running inference", unit="batch"):
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
            preds = get_binary_segmentation_predictions(model, imgs)

            for i in range(imgs.size(0)):
                img_path     = test_dataset.image_paths[len(all_samples)]
                label_path   = test_dataset.label_paths[len(all_samples)]
                pulse        = extract_pulse(img_path)
                filename     = os.path.basename(img_path)

                img_cpu      = imgs[i].cpu()
                gt_cpu       = masks[i].cpu()
                pred_cpu     = preds[i].cpu()
                orig_img_cpu = orig_imgs[i].cpu()

                all_samples.append((img_cpu, gt_cpu, pred_cpu, orig_img_cpu, img_path, label_path))

                # === Unified metric inputs ===
                # With recovery
                pred_tensor = TF.to_tensor(recover_original(TF.to_pil_image(pred_cpu.byte() * 255))).squeeze(0).long()
                gt_tensor   = TF.to_tensor(recover_original(TF.to_pil_image(gt_cpu.byte() * 255))).squeeze(0).long()

                # Without recovery
                # pred_tensor = pred_cpu.long()
                # gt_tensor   = gt_cpu.long()

                # === Compute metrics ===
                metrics = calculate_all_metrics(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))
                metrics['Pulse'] = pulse
                metrics['Filename'] = filename

                # === Compute ablation area in mm² ===
                pixel_area_mm2 = 0.001
                metrics['pred_area_mm2'] = compute_area(pred_tensor) * pixel_area_mm2
                metrics['gt_area_mm2']    = compute_area(gt_tensor) * pixel_area_mm2
                individual_metrics.append(metrics)

                # === Save recovered prediction ===
                match = re.match(r't3US(\d+)_(\d+)_(\d+)', filename)
                if match:
                    pred_filename = f"t3Label{match.group(1)}_{match.group(2)}_{match.group(3)}.png"
                    recovered_pred_pil = recover_original(TF.to_pil_image(pred_cpu.byte() * 255))
                    recovered_pred_pil.save(os.path.join(pred_dir, pred_filename))



    print(f"Total samples = {len(all_samples)}")

    # Save per-image metrics
    indiv_metrics_csv = os.path.join("test_results", config.EXPERIMENT_NAME, "individual_metrics.csv")
    pd.DataFrame(individual_metrics).to_csv(indiv_metrics_csv, index=False)
    print(f"Saved per-image metrics to {indiv_metrics_csv}")

    # --- Aggregate Metrics ---
    # Without recovery
    # all_preds = torch.stack([s[2].squeeze(0) for s in all_samples])   # [B, H, W]
    # all_targets = torch.stack([s[1] for s in all_samples])            # [B, H, W]

    # With Recovery
    all_preds = torch.stack([
        TF.to_tensor(recover_original(TF.to_pil_image(s[2].byte() * 255))).squeeze(0).long()
        for s in all_samples
    ])

    all_targets = torch.stack([
        TF.to_tensor(recover_original(TF.to_pil_image(s[1].byte() * 255))).squeeze(0).long()
        for s in all_samples
    ])


    final_metrics = calculate_all_metrics(all_preds, all_targets)
    print(final_metrics)
    print("\n--- Average Test Metrics ---")
    if final_metrics:
        for k, v in sorted(final_metrics.items()):
            print(f"{k}: {v:.4f}" if isinstance(v, (float, np.number)) and pd.notna(v) else f"{k}: {v}")

        summary_csv_path = os.path.join("test_results", config.EXPERIMENT_NAME, "test_summary_metrics.csv")
        log_test_metrics_to_csv(summary_csv_path, config, final_metrics)

        plot_metrics_vs_pulses(indiv_metrics_csv, vis_dir, config.EXPERIMENT_NAME)
        print("\n--- Plotting Ablation Area ---")

        # Load CSV
        df = pd.read_csv(indiv_metrics_csv)

        # Parse pulse and dataset from filename
        def extract_pulse_and_dataset(filename):
            base = os.path.basename(filename).replace(".jpg", "").replace(".png", "")
            parts = base.split('_')
            dataset = int(parts[-1]) if parts[-1].isdigit() else -1
            pulse_match = re.search(r'US(\d+)', base)
            pulse = int(pulse_match.group(1)) if pulse_match else -1
            return pulse, dataset

        df[['pulse', 'dataset']] = df['Filename'].apply(lambda x: pd.Series(extract_pulse_and_dataset(x)))

        # Group by pulse & dataset
        grouped_df = df.groupby(['pulse', 'dataset'])[['gt_area_mm2', 'pred_area_mm2']].mean().reset_index()

        # Save grouped data
        grouped_csv_path = os.path.join("test_results", config.EXPERIMENT_NAME, "area_grouped_by_pulse_dataset.csv")
        grouped_df.to_csv(grouped_csv_path, index=False)
        print(f"✅ Saved grouped CSV to {grouped_csv_path}")

        # Plot: GT vs Predicted Area vs Pulses
        pulse_agg = grouped_df.groupby('pulse')[['gt_area_mm2', 'pred_area_mm2']].agg(['mean', 'std']).reset_index()
        pulse_agg.columns = ['pulse', 'gt_mean', 'gt_std', 'pred_mean', 'pred_std']
        pulse_agg['pulse'] = pulse_agg['pulse'].astype(int) * 20  # scale pulse index if needed

        plt.figure(figsize=(12, 6))
        plt.plot(pulse_agg['pulse'], pulse_agg['gt_mean'], label='Ground Truth Mean', color='orange')
        plt.fill_between(pulse_agg['pulse'],
                        pulse_agg['gt_mean'] - pulse_agg['gt_std'],
                        pulse_agg['gt_mean'] + pulse_agg['gt_std'],
                        alpha=0.3, color='orange', label='GT ± Std')

        plt.plot(pulse_agg['pulse'], pulse_agg['pred_mean'], label='Predicted Mean', color='blue')
        plt.fill_between(pulse_agg['pulse'],
                        pulse_agg['pred_mean'] - pulse_agg['pred_std'],
                        pulse_agg['pred_mean'] + pulse_agg['pred_std'],
                        alpha=0.3, color='blue', label='Prediction ± Std')

        plt.xlabel('Number of Pulses')
        plt.ylabel('Ablation Area (mm²)')
        plt.title('GT vs Predicted Ablation Area vs Pulses')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join("test_results", config.EXPERIMENT_NAME, "ablation_area_vs_pulses.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✅ Saved manual area plot to {plot_path}")



    # --- Visualization Grid ---
    for sample_idx, (img, gt, pred, orig_img, img_path, label_path) in tqdm(enumerate(all_samples), total=len(all_samples), desc="Saving visualizations", unit="img"):
        orig_img = Image.open(img_path).convert("RGB")
        pred_pil = TF.to_pil_image(pred.byte() * 255)
        gt_pil = TF.to_pil_image(gt.byte() * 255)

        pred_recovered = recover_original(pred_pil)
        gt_recovered = recover_original(gt_pil)

        # Without recovery
        # images = [orig_img, pred_pil, gt_pil]

        # With recovery
        images = [orig_img, pred_recovered, gt_recovered]

        pulse = extract_pulse(img_path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['B-Mode Image', 'Predicted Mask', 'Ground Truth']

        for col_idx, (ax, im) in enumerate(zip(axes, images)):
            ax.imshow(im, cmap='gray' if im.mode == 'L' else None)
            ax.axis('off')
            ax.set_title(titles[col_idx], fontsize=14)
            if col_idx == 0:
                ax.text(-0.1, 0.5, f'No. of Pulses: {pulse}', transform=ax.transAxes, fontsize=14, ha='center', va='center', rotation=90)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        filename = os.path.splitext(os.path.basename(label_path))[0] + "_vis.png"
        plt.savefig(os.path.join(vis_dir, filename), bbox_inches='tight', dpi=150)
        plt.close(fig)

    print("Visualizations saved.")
    print("\n--- Testing Finished ---")

if __name__ == "__main__":
    main()