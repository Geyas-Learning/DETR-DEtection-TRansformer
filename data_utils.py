"""
=========================================================
DATA UTILITIES — for DETR spacecraft model
=========================================================
"""
import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from os.path import join, isfile
from torch import nn, Tensor
from typing import List, Tuple
from util.misc import nested_tensor_from_tensor_list


# ✅ Base data directory
DATA_ROOT = os.path.join(".", "data")

CLASS_NAMES = [
    "VenusExpress", "Cheops", "LisaPathfinder", "ObservationSat1",
    "Proba2", "Proba3", "Proba3ocs", "Smart1", "Soho", "XMM Newton"
]
NUM_CLASSES = len(CLASS_NAMES)
# Mapping from string name to integer class ID (0 to 9)
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}


# ============================================================
# COLLATE FUNCTION
# ============================================================
def detr_collate_fn(batch: List[Tuple[Tensor, dict]]):
    """
    Collate function specialized for the DETR DataLoader.
    
    1. Batches input images into a NestedTensor (padding images to the max size 
       in the batch, along with a mask indicating the original content).
    2. Keeps targets (boxes, labels) as a list of dictionaries.
    """
    # Separate images and targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # utility function to create the NestedTensor (from util.misc)
    nested_tensor = nested_tensor_from_tensor_list(images) 
    return nested_tensor, targets


# ============================================================
# DATASET CLASS (for labeled data)
# ============================================================
class SpacecraftDataset(Dataset):
    """
    Loads labeled spacecraft images for DETR training/validation.
    
    The output format is: (image_tensor, target_dict)
    - image_tensor: Normalized [C, H, W] tensor (e.g., [3, 224, 224]).
    - target_dict: {'boxes': normalized cxcywh tensor, 'labels': class IDs, 'orig_size': H, W}
    """
    def __init__(self, csv_file, root_dir=DATA_ROOT, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        # Processed CSV with columns: image_path, xmin, ymin, xmax, ymax, class_id
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing image: {img_path}")

        full_h, full_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Hardcoded resize for simplicity/consistency (224x224)
        img_resized = cv2.resize(img, (224, 224))

        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        class_id = row["class_id"]
        
        # --- DETR Target Preparation (Normalized cxcywh) ---
        # 1. Calculate center coordinates
        x_center = ((xmin + xmax) / 2) / full_w
        y_center = ((ymin + ymax) / 2) / full_h
        # 2. Calculate width and height
        bw = (xmax - xmin) / full_w
        bh = (ymax - ymin) / full_h

        # Bounding box in cxcywh normalized format (N, 4) where N=1
        bbox_norm = torch.tensor([x_center, y_center, bw, bh], dtype=torch.float32).unsqueeze(0)
        
        # Labels must be LongTensor (size N)
        labels = torch.tensor([class_id], dtype=torch.long) 

        # Image Tensor (C, H, W) normalized [0, 1]
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Create the DETR target dictionary
        target_dict = {
            'boxes': bbox_norm,
            'labels': labels,
            'orig_size': torch.as_tensor([full_h, full_w]), 
            'size': torch.as_tensor([img_resized.shape[0], img_resized.shape[1]]) 
        }

        return img_tensor, target_dict


# ============================================================
# TEST DATASET CLASS (for inference)
# ============================================================
class TestSpacecraftDataset(Dataset):
    """
    Loads raw images from a directory for inference.
    Returns: (image_tensor, image_name, original_height, original_width)
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_names = sorted([
            f for f in os.listdir(img_dir)
            if isfile(join(img_dir, f)) and f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not read image: {img_path}")
            # Return dummy tensor on failure to prevent crash
            return torch.empty((3, 224, 224), dtype=torch.float32), img_name, 0, 0

        full_h, full_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Must match the training size
        img_resized = cv2.resize(img, (224, 224))
        # Image Tensor (C, H, W) normalized [0, 1]
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Return original dimensions for post-processing and rescaling of predicted boxes
        return img_tensor, img_name, full_h, full_w


# ============================================================
# CSV PREPROCESSING
# ============================================================
def preprocess_original_csv(csv_path, output_csv, split="train"):
    """
    Converts original annotation CSV (from challenge) into a simplified, 
    model-ready format containing bounding box coordinates and class IDs.
    
    Input columns: [Image name, Class, Bounding box]
    Output columns: [image_path, xmin, ymin, xmax, ymax, class_id]
    """
    print(f"[INFO] Preprocessing {csv_path} → {output_csv}")
    df = pd.read_csv(os.path.join(DATA_ROOT, csv_path))
    new_rows = []

    for _, row in df.iterrows():
        image_name = row["Image name"]
        cls_name = row["Class"]
        bbox_str = row["Bounding box"]
        # Skip if the class name is not one of the 10 target classes
        if cls_name not in CLASS_MAP:
            continue
        try:
            # Parse the bounding box string "(x1, y1, x2, y2)"
            xmin, ymin, xmax, ymax = [int(x.strip()) for x in bbox_str.strip("()").split(",")]
        except Exception:
            print(f"[WARN] Skipping malformed bbox: {bbox_str}")
            continue
        
        # Construct the relative path for easy lookup in the data directory
        image_path = os.path.join("images", cls_name, split, image_name)
        new_rows.append({
            "image_path": image_path,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "class_id": CLASS_MAP[cls_name]
        })

    # Save the cleaned and processed data to the output CSV
    pd.DataFrame(new_rows).to_csv(output_csv, index=False)
    print(f"[INFO] ✅ Saved processed CSV → {output_csv} ({len(new_rows)} rows)")


# ============================================================
# VALIDATION
# ============================================================
def validate_image_paths(csv_path):
    """
    Checks the specified processed CSV for file paths and reports how many 
    images are valid (exist) and how many are missing.
    """
    print(f"[CHECK] Validating image paths from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    valid, missing = 0, 0
    for _, row in df.iterrows():
        path = os.path.join(DATA_ROOT, row["image_path"])
        if os.path.exists(path):
            valid += 1
        else:
            missing += 1
    print(f"[CHECK] ✅ Valid: {valid} | ⚠️ Missing: {missing}")
    return valid, missing