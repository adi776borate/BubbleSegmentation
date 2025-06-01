# train.py
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import pandas as pd
import warnings
import csv
from datetime import datetime
from config import Config
from model import *
from loss import * 
from metric import calculate_all_metrics
from dataloader import create_ultrasound_dataloaders
import torchvision.transforms.functional as TF

from PIL import Image, ImageOps
import re
from glob import glob
from torch.utils.data import DataLoader

from config import Config
from model import *
from loss import *
from metric import calculate_all_metrics
from dataloader import BubbleDataset
from utils import get_binary_segmentation_predictions


# Suppress specific warnings if needed
warnings.filterwarnings("ignore")

def get_model(config):
    device = config.DEVICE
    model_name = config.MODEL_NAME
    in_channels = config.IN_CHANNELS
    num_classes = config.NUM_CLASSES
    encoder_name = config.ENCODER_NAME if hasattr(config, 'ENCODER_NAME') else 'resnet34'
    encoder_weights = config.ENCODER_WEIGHTS if hasattr(config, 'ENCODER_WEIGHTS') else 'imagenet'
    use_cuda = config.USE_CUDA if hasattr(config, 'USE_CUDA') else True
    
    print(f"--- Initializing Model: {model_name} ---")
    print(f"Input Channels: {in_channels}, Num Classes: {num_classes}")
    
    if model_name == "ResNet18CNN":
        model = resnet18(
            in_channels=in_channels,
            num_classes=num_classes,
        )
        print(f"ResNet18CNN pretrained loaded .. ")
        
    elif model_name == "DeepLabV3Plus":
        model = deeplabv3p(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
        print(f"DeepLabV3Plus model initialized with encoder {encoder_name}")
    
    elif model_name == "FPN":
        model = fpn(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
        print(f"FPN model initialized with encoder {encoder_name}")
    
    elif model_name == "TorchvisionDeepLabV3":
        model = torchvision_deeplabv3(
            num_classes=num_classes,
        )
        print(f"TorchvisionDeepLabV3 model initialized")
    
    elif model_name == "Unet++":
        model = unetpp(
            in_channels=in_channels,
            classes=num_classes
        )
        print(f"Unet++ model initialized")
    
    elif model_name == "MaskRCNN":
        model = maskrcnn(
            in_channels=in_channels,
            num_classes=num_classes
        )
        print(f"MaskRCNN model loaded")
    
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    return model.to(device)

def get_loss_fn(config):
    """Initializes the loss function based on the configuration."""
    loss_fn_name = config.LOSS_FN
    print(f"--- Initializing Loss Function: {loss_fn_name} ---")

    if loss_fn_name == "DiceFocalLoss":
        dice_weight = getattr(config, 'LOSS_DICE_WEIGHT', 0.5)
        focal_weight = getattr(config, 'LOSS_FOCAL_WEIGHT', 0.6)
        criterion = DiceFocalLoss(dice_weight=dice_weight, focal_weight=focal_weight)
        print(f"DiceFocalLoss Params - Dice Weight: {dice_weight}, Focal Weight: {focal_weight}")

    elif loss_fn_name == "DiceLoss":
        criterion = DiceLoss()
        print(f"DiceLoss")
    
    elif loss_fn_name == "DiceFocalWithPulsePriorLoss":
        criterion = DiceFocalWithPulsePriorLoss()
        print(f"DiceFocalWithPulsePriorLoss")

    elif loss_fn_name == "AsymmetricFocalTverskyLoss":
        tversky_weight = getattr(config, 'LOSS_TVERSKY_WEIGHT', 0.5)
        focal_weight = getattr(config, 'LOSS_FOCAL_WEIGHT', 0.6)
        criterion = AsymmetricFocalTverskyLoss(tversky_weight=tversky_weight, focal_weight=focal_weight)
        print(f"AsymmetricFocalTverskyLoss Params - Tversky Weight: {tversky_weight}, Focal Weight: {focal_weight}")
    else:
        raise ValueError(f"Invalid loss function name in config: '{loss_fn_name}'")

    print("--- Loss Function Initialized ---")
    return criterion

def train_one_epoch(model, optimizer, criterion, train_loader, epoch, config, writer):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Train)")
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, batch_data in enumerate(loop):
        if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 5:
            continue

        data, targets, _, image_paths,_ = batch_data
        data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)  # Ensure shape [B, 1, H, W]

        # Extract pulse numbers
        def extract_pulse_number(path):
            import re
            match = re.search(r'US(\d+)', os.path.basename(path))
            if not match:
                raise ValueError(f"Pulse number could not be extracted from path: {path}")
            return int(match.group(1))

        pulses = torch.tensor([extract_pulse_number(p) for p in image_paths], dtype=torch.float32).to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(data)
        if isinstance(outputs, dict) and 'out' in outputs:
            logits = outputs['out']
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            raise ValueError(f"Unsupported model output type: {type(outputs)}")

        # Use pulse if required by loss
        if isinstance(criterion, DiceFocalWithPulsePriorLoss):
            loss = criterion(logits, targets, pulses)
        else:
            loss = criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    return avg_loss
    
def validate_one_epoch(model, criterion, val_loader, epoch, config, writer):
    import re
    model.eval()
    loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Validation)")
    batch_metrics_list = []
    total_val_loss = 0.0
    num_batches = len(val_loader)

    def extract_pulse_number(path):
        match = re.search(r'US(\d+)', os.path.basename(path))
        if not match:
            raise ValueError(f"Pulse number could not be extracted from path: {path}")
        return int(match.group(1))

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loop):
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 5:
                continue

            data, targets, _, image_paths, _ = batch_data
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

            # Ensure targets are [B, 1, H, W] for loss
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)
            targets = (targets > 0).long()

            pulses = torch.tensor(
                [extract_pulse_number(p) for p in image_paths],
                dtype=torch.float32,
            ).to(config.DEVICE)

            outputs = model(data)
            if isinstance(outputs, dict) and 'out' in outputs:
                logits = outputs['out']
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                raise ValueError(f"Unsupported model output type: {type(outputs)}")

            if isinstance(criterion, DiceFocalWithPulsePriorLoss):
                loss = criterion(logits, targets, pulses)
            else:
                loss = criterion(logits, targets)

            total_val_loss += loss.item()

            # Metric computation: strip channel dim from targets
            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            # Pass raw logits and clean targets to the metric function
            batch_metrics = calculate_all_metrics(logits, targets)
            batch_metrics_list.append(batch_metrics)

            loop.set_postfix(loss=loss.item())

    if not batch_metrics_list:
        return total_val_loss / max(num_batches, 1), {}

    metrics_df = pd.DataFrame(batch_metrics_list)
    avg_metrics_dict = metrics_df.mean(axis=0).to_dict()
    avg_val_loss = total_val_loss / num_batches

    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    for key, value in avg_metrics_dict.items():
        if pd.notna(value) and isinstance(value, (float, int, np.number)):
            writer.add_scalar(f"Metrics/{key}", value, epoch)

    return avg_val_loss, avg_metrics_dict


def save_checkpoint(model, optimizer, filename):
    """Saves checkpoint."""
    try:
        print(f"=> Saving checkpoint to {filename}")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(checkpoint, filename)
    except Exception as e:
        print(f"Error saving checkpoint to {filename}: {e}")

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """Loads checkpoint."""
    if not os.path.isfile(checkpoint_file):
        print(f"=> Checkpoint file not found at {checkpoint_file}. Skipping load.")
        return
    print(f"=> Loading checkpoint from {checkpoint_file}")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        # model.load_state_dict(torch.load("best_model_final.pth"))
        if optimizer is not None and "optimizer" in checkpoint:
             optimizer.load_state_dict(checkpoint["optimizer"])
             for param_group in optimizer.param_groups:
                 param_group["lr"] = lr # Reset LR from config
        print("=> Checkpoint loaded successfully")
    except Exception as e:
        print(f"=> Error loading checkpoint: {e}")
        return


# --- NEW: CSV Logging Function ---
def log_metrics_to_csv(log_path, epoch, config, train_loss, val_loss, metrics_dict):
    """Appends metrics and config details for an epoch to a CSV file."""
    file_exists = os.path.isfile(log_path)
    # Define header including essential config and all metric keys
    header = [
        'Timestamp', 'Epoch', 'Experiment_Name', 'Model_Name', 'Optimizer', 
        'Loss_Function','Learning_Rate', 'Batch_Size',
        'Train_Loss', 'Validation_Loss'
    ]
    # Dynamically add metric keys from the dictionary, ensuring order
    metric_keys = sorted([key for key in metrics_dict.keys() if pd.notna(metrics_dict[key])]) # Filter out potential NaNs
    header.extend(metric_keys)

    # Prepare data row, converting values to strings for CSV
    data_row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Epoch': epoch + 1, # Use 1-based epoch for logging
        'Experiment_Name': config.EXPERIMENT_NAME,
        'Model_Name': config.MODEL_NAME,
        'Optimizer': config.OPTIMIZER,
        'Loss_Function': config.LOSS_FN,
        'Learning_Rate': f"{config.LEARNING_RATE:.1E}", # Scientific notation
        'Batch_Size': config.BATCH_SIZE,
        'Train_Loss': f"{train_loss:.6f}",
        'Validation_Loss': f"{val_loss:.6f}"
    }
    # Add formatted metrics only for the valid keys
    for key in metric_keys:
         value = metrics_dict[key]
         data_row[key] = f"{value:.6f}" if isinstance(value, (float, np.number)) else str(value)

    # Write to CSV
    try:
        with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction='ignore') # Ignore extra keys not in header
            if not file_exists or os.path.getsize(log_path) == 0: # Check size too
                writer.writeheader() # Write header only if file is new or empty
            writer.writerow(data_row)
    except IOError as e:
        print(f"Error writing to CSV log file {log_path}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during CSV logging: {e}")


def main():
    config = Config() # Load configuration

    train_loader, val_loader = create_ultrasound_dataloaders()

    # --- Setup Directories ---
    experiment_dir = os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME)
    model_ckpt_dir = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME)
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    # --- CSV Log File Path ---
    csv_log_path = os.path.join(experiment_dir, config.CSV_LOG_FILE)
    print(f"CSV metrics log will be saved to: {csv_log_path}")

    if not hasattr(config, 'VISUALIZE_EVERY'):
        config.VISUALIZE_EVERY = 2
    if not hasattr(config, 'SAVE_MODEL'): # Add default if missing
        config.SAVE_MODEL = True


    # --- Initialize Model, Loss, Optimizer ---
    model = get_model(config)
    criterion = get_loss_fn(config)
    # optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    if config.OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.SGD_MOMENTUM
        )
        if config.USE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        else:
            scheduler = None
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        if config.USE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        else:
            scheduler = None

    # --- TensorBoard Writer ---
    writer = SummaryWriter(log_dir=experiment_dir)
    print(f"TensorBoard logs will be saved in: {experiment_dir}")
    print(f"Checkpoints will be saved in: {model_ckpt_dir}")


    # --- Training Loop ---
    best_val_loss = float('inf')
    best_val_iou = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, epoch, config, writer)
        val_loss, avg_val_metrics = validate_one_epoch(model, criterion, val_loader, epoch, config, writer)

        if scheduler is not None:
            val_iou = avg_val_metrics.get("MeanIoU", 0.0) if avg_val_metrics else 0.0
            scheduler.step(val_iou)

        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Check if metrics dictionary is not empty before printing/logging
        if avg_val_metrics:
            print("Average Validation Metrics:")
            for key, value in avg_val_metrics.items():
                # Handle potential non-numeric values gracefully during print
                try: print(f"  {key}: {float(value):.4f}")
                except (ValueError, TypeError): print(f"  {key}: {value}")

            # --- Log Metrics to CSV --- (Moved inside the check)
            log_metrics_to_csv(csv_log_path, epoch, config, train_loss, val_loss, avg_val_metrics)
        else:
             print("Validation metrics could not be calculated for this epoch.")



        # Save checkpoint periodically (e.g., every 2 epochs)
        if (epoch + 1) % 2 == 0 :
            ckpt_path = os.path.join(model_ckpt_dir, f"epoch_{epoch+1}.pth.tar")
            save_checkpoint(model, optimizer, filename=ckpt_path)

        # --- Visualize Predictions ---
        if (epoch + 1) % config.VISUALIZE_EVERY == 0:
            visualize_predictions(model, val_loader, config, epoch, writer, num_samples=10)

        val_iou  = avg_val_metrics["MeanIoU"]

        if val_iou > best_val_iou : 
            ckpt_path = os.path.join(model_ckpt_dir, f"best.pth.tar")
            best_val_iou = val_iou 
            save_checkpoint(model, optimizer, filename=ckpt_path)

    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final evaluation...")
        model.load_state_dict(torch.load(best_model_path))

    writer.close()
    print("--- Training Finished ---")
    print(f"CSV log saved: {csv_log_path}")


def recover_original(img, final_width=1024, final_height=256, cropped_width=750, cropped_height=554):
    img_resized = TF.resize(img, [cropped_height, cropped_width], interpolation=Image.NEAREST)
    pad_left = (final_width - cropped_width) // 2
    pad_right = final_width - cropped_width - pad_left
    pad_top = (final_height - cropped_height) // 2
    pad_bottom = final_height - cropped_height - pad_top
    return ImageOps.expand(img_resized, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def visualize_predictions(model, val_loader, config, epoch, writer, num_samples=10):
    import re
    model.eval()
    images_shown = 0
    save_dir = os.path.join("logs", config.EXPERIMENT_NAME, "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    # --- Shared pulse extractor ---
    def extract_pulse_number(path):
        base = os.path.basename(path)
        match = re.search(r'US(\d+)', base)
        if not match:
            raise ValueError(f"Pulse number could not be extracted from path: {path}")
        return int(match.group(1))

    with torch.no_grad():
        for batch_data in val_loader:
            if images_shown >= num_samples:
                break
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 5:
                continue

            data, targets, _, image_paths,_ = batch_data
            data = data.to(config.DEVICE)

            preds_binary = get_binary_segmentation_predictions(model, data).cpu()

            for i in range(data.size(0)):
                if images_shown >= num_samples:
                    break

                img_tensor = data[i].cpu()
                pred_tensor = preds_binary[i].cpu()
                gt_tensor = targets[i].cpu()
                img_path = image_paths[i]  # ✅ Always extract pulse from image path

                input_img = TF.to_pil_image(img_tensor)
                pred_pil = TF.to_pil_image(pred_tensor.byte() * 255)
                gt_pil = TF.to_pil_image(gt_tensor.byte() * 255)

                pred_restored = recover_original(pred_pil)
                gt_restored = recover_original(gt_pil)
                input_restored = recover_original(input_img.convert("L"))

                pulse = extract_pulse_number(img_path)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for idx, (ax, im) in enumerate(zip(axes, [input_restored, pred_restored, gt_restored])):
                    ax.imshow(im, cmap='gray' if im.mode == 'L' else None)
                    ax.axis('off')
                    if idx == 0:
                        ax.text(-0.1, 0.5, f'Pulse: {pulse}', transform=ax.transAxes,
                                fontsize=14, ha='center', va='center', rotation=90)
                axes[0].set_title("B-Mode")
                axes[1].set_title("Predicted Mask")
                axes[2].set_title("Ground Truth")

                filename = os.path.basename(img_path)
                save_path = os.path.join(save_dir, f"epoch{epoch+1:02d}_sample{images_shown:03d}_{filename}")
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                images_shown += 1

if __name__ == "__main__":
    main()