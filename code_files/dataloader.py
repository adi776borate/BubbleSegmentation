# STEP 0: Imports
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision.transforms import functional as TF
from config import Config

# STEP 1: Custom Dataset for Bubble Images
class BubbleDataset(Dataset):
    def __init__(self, image_paths, label_paths, augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def center_crop(self, img, target_width=750, target_height=554):
        w, h = img.size
        left = (w - target_width) // 2
        top = (h - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        return img.crop((left, top, right, bottom))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert('L')
        label = Image.open(label_path)

        # original_image_tensor = TF.to_tensor(image).expand(3, -1, -1)
        original_image_tensor = TF.to_tensor(image)

        # Center crop
        image = self.center_crop(image, target_width=750)
        label = self.center_crop(label, target_width=750)

        image = TF.resize(image, (256, 256))
        label = TF.resize(label, (256, 256), interpolation=Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
            if random.random() > 0.5:
                angle = random.uniform(-5, 5)
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle, interpolation=Image.NEAREST)

            # Brightness / Contrast
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))

            # Random Crop & Resize (mild zoom)
            if random.random() > 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.9, 1.0), ratio=(1.0, 1.0))
                image = TF.resized_crop(image, i, j, h, w, (256, 256))
                label = TF.resized_crop(label, i, j, h, w, (256, 256), interpolation=Image.NEAREST)

            # Gaussian Noise
            if random.random() > 0.5:
                img_tensor = TF.to_tensor(image)
                noise = torch.randn_like(img_tensor) * 0.01
                img_tensor = (img_tensor + noise).clamp(0, 1)
                image = TF.to_pil_image(img_tensor)

        image = TF.to_tensor(image)
        image = image.expand(3, -1, -1)
        label = TF.pil_to_tensor(label).squeeze().long()
        label = (label > 127).long()

        return image, label, original_image_tensor, image_path, label_path


def extract_dataset_number(path):
    return int(path.split('_')[-1].split('.')[0])


def create_ultrasound_dataloaders():
    config = Config() # Load configuration
    # STEP 2: Parsing and Splitting Data Based on Dataset Number
    all_images = sorted(glob.glob('../Data/US_2/*.jpg'))
    all_labels = [img_path.replace('US', 'Label').replace('.jpg', '.png') for img_path in all_images]

    groups = [extract_dataset_number(p) for p in all_images]

    splitter = GroupShuffleSplit(n_splits=1, test_size=1/6)
    print("Number of unique groups:", len(np.unique(groups)))
    train_idx, val_idx = next(splitter.split(all_images, groups=groups))

    train_images = [all_images[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_images = [all_images[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    print(np.array(set([extract_dataset_number(p) for p in train_images])))
    print(np.array(set([extract_dataset_number(p) for p in val_images])))

    print("Sample mapping:")
    for img, lbl in zip(train_images[:3], train_labels[:3]):
        print(f"{img}  -->  {lbl}")

    train_dataset = BubbleDataset(train_images, train_labels, augment=True)
    val_dataset = BubbleDataset(val_images, val_labels, augment=False)

    train_datasets = sorted(set(extract_dataset_number(p) for p in train_images))
    val_datasets = sorted(set(extract_dataset_number(p) for p in val_images))

    print("Train dataset numbers:", train_datasets)
    print("Validation dataset numbers:", val_datasets)

    # STEP 4: Loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE*2, shuffle=False)

    # STEP 5: Check output of new dataset 
    img_batch, lbl_batch, orig_img_batch, _, _ = next(iter(train_loader))

    print("Image shape:", img_batch.shape)
    print("Label shape:", lbl_batch.shape)
    print("Original Image shape:", orig_img_batch.shape)

    print("Label dtype:", lbl_batch.dtype)
    print("Label values:", lbl_batch.unique())

    return train_loader , val_loader
if __name__ == "__main__":
    train_loader, val_loader = create_ultrasound_dataloaders()