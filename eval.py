import os
import torch
import cv2
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple

# =======================================================================
# CRITICAL IMPORTS/CLASSES RECONSTRUCTED FROM YOUR CODE SNIPPET
# NOTE: These assume 'data_utils.py', 'detr_model', and 'util' modules are accessible
# =======================================================================

# --- Configuration Constants (Must match your data_utils.py) ---
# Assuming DATA_ROOT is defined in your environment
DATA_ROOT = os.path.join(".", "data")

CLASS_NAMES = [
    "VenusExpress", "Cheops", "LisaPathfinder", "ObservationSat1",
    "Proba2", "Proba3", "Proba3ocs", "Smart1", "Soho", "XMM Newton"
]
NUM_CLASSES = len(CLASS_NAMES)


# --- DETR Args Class (Necessary for build_model) ---
class Args:
    """Simple class to hold required DETR configuration arguments."""
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.dataset_file = 'custom' 
        self.frozen_weights = None
        self.num_classes = config.get("num_classes", NUM_CLASSES + 1) 
        self.backbone = 'resnet50' 
        self.dilation = True
        self.position_embedding = 'sine'
        self.masks = config.get("masks", False) 
        self.num_queries = 100 
        self.aux_loss = True
        self.lr_backbone = 1e-5 
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 6
        self.dec_layers = 6
        self.pre_norm = False
        self.clip_max_norm = 0.1 
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.3 

# --- Utility Functions (Must be imported or defined here) ---
# NOTE: These lines assume your local setup allows importing these modules.
from data_utils import SpacecraftDataset # Assuming this is your dataset class
from detr_model.__init__ import build_model # Assuming this is the model constructor
from util.misc import nested_tensor_from_tensor_list # Assuming this utility exists
from util.box_ops import box_cxcywh_to_xyxy # Assuming this utility exists

# ============================================================
# UTILITY: Denormalize Boxes 
# ============================================================
def rescale_box(bbox, size):
    """
    Rescales normalized [0, 1] xyxy boxes to pixel values.
    'size' must be a tensor of shape (2,) representing [H, W].
    """
    # FIX: size.tolist() will work correctly here because 'size' is guaranteed to be shape (2,)
    h, w = size.tolist()
    scale_fct = torch.tensor([w, h, w, h], device=bbox.device)
    scaled_bbox = bbox * scale_fct
    return scaled_bbox

# ============================================================
# VISUALIZATION FUNCTION (GT vs Prediction)
# ============================================================
def visualize_validation_comparison(config, num_samples=100):
    print(f"\n[Visualizer] 🖼️ Starting GT vs Prediction visualization for {num_samples} samples...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup Directories ---
    output_dir = os.path.join(config["project_dir"], f"val_visualized_{num_samples}")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Load Model and Post-processor ---
    detr_args = Args(config)
    detr_args.num_classes = NUM_CLASSES + 1 
    model, _, postprocessors = build_model(detr_args)
    model.to(device)
    
    model_path = os.path.join(config["project_dir"], "weights", f"{config['run_name']}_best.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model weights not found: {model_path}. Run training first.")
        return

    # Load the weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle both full checkpoint (if you updated your save logic) or just state_dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
         model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    print(f"✅ Loaded weights from: {model_path}")

    # --- 2. Load Validation Data ---
    val_data = SpacecraftDataset(config["val_csv"], DATA_ROOT)
    
    # --- Generate Indices ---
    indices_to_visualize = torch.randperm(len(val_data))[:num_samples].tolist() 
    print(f"Loaded {len(indices_to_visualize)} indices for visualization.")
    
    # --- 3. Inference and Visualization Loop (FIXED) ---
    
    with torch.no_grad():
        for i, global_idx in enumerate(indices_to_visualize):
            
            # 1. Directly fetch the single sample from the dataset
            try:
                img_tensor, target_dict = val_data[global_idx]
            except IndexError:
                print(f"[WARN] Index {global_idx} out of bounds, skipping.")
                continue

            # 2. Convert single sample into DETR batch format
            samples = nested_tensor_from_tensor_list([img_tensor]).to(device)
            targets = [{k: v.to(device) for k, v in target_dict.items()}]
            
            # Unpack target data for this single image
            target = targets[0]
            
            # --- Load Original Image (for visualization) ---
            csv_row = val_data.data.iloc[global_idx]
            img_path = os.path.join(DATA_ROOT, csv_row["image_path"])
            img_name = os.path.basename(img_path)
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not load image: {img_path}")
                continue
            orig_h, orig_w = img.shape[:2]
                
            # --- Ground Truth (GT) Boxes ---
            gt_boxes_norm_cxcywh = target['boxes'].squeeze(0) # (4,) normalized cxcywh
            gt_boxes_norm_xyxy = box_cxcywh_to_xyxy(gt_boxes_norm_cxcywh)
            
            # --- FIX: Ensure orig_size tensor has shape (2,) for rescaling ---
            orig_size_correct_shape = target['orig_size']
            
            # Check if the tensor has an extra dimension (e.g., shape (1, 2)) and squeeze it
            if orig_size_correct_shape.dim() > 1:
                 orig_size_correct_shape = orig_size_correct_shape.squeeze(0)

            # Rescale the normalized box (1, 4) using the (2,) size tensor
            gt_boxes_xyxy = rescale_box(
                gt_boxes_norm_xyxy.unsqueeze(0), 
                orig_size_correct_shape
            ).cpu().numpy()
            
            gt_labels = target['labels'].cpu().numpy()
            
            # --- Model Prediction Boxes ---
            outputs = model(samples)
            
            # The DETR postprocessors expect the target sizes tensor to be (N, 2), where N=1 here.
            orig_target_sizes = orig_size_correct_shape.unsqueeze(0).to(device) 
            
            results_post = postprocessors['bbox'](outputs, orig_target_sizes)[0]
            
            # Keep predictions with score > threshold (e.g., 70% confidence)
            prob_threshold = 0.7
            keep = results_post['scores'] > prob_threshold
            
            pred_boxes_xyxy = results_post['boxes'][keep].cpu().numpy()
            pred_labels = results_post['labels'][keep].cpu().numpy()
            pred_scores = results_post['scores'][keep].cpu().numpy()
            
            
            # --- Draw Boxes ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            
            # 1. Draw GT Boxes (Green)
            for box, label_id in zip(gt_boxes_xyxy, gt_labels):
                xmin, ymin, xmax, ymax = box.round().astype(int)
                class_name = CLASS_NAMES[label_id]
                color = (0, 255, 0) # Green for Ground Truth
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                
                label = f"GT: {class_name}"
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                # Draw text box at the top
                cv2.rectangle(img, (xmin, max(0, ymin - text_height - 4)), (xmin + text_width, ymin), color, -1)
                cv2.putText(img, label, (xmin, max(10, ymin - 2)), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            # 2. Draw Prediction Boxes (Red)
            for box, label_id, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                xmin, ymin, xmax, ymax = box.round().astype(int)
                
                # Clamp coordinates
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(orig_w, xmax), min(orig_h, ymax)
                
                if label_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[label_id]
                    color = (0, 0, 255) # Red for Prediction
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    label = f"Pred: {class_name} ({score:.2f})"
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    
                    # Try to place the label below the bounding box
                    text_y = ymax + text_height + 4
                    if text_y > orig_h: # If it goes off the bottom, place it inside or slightly above
                        text_y = ymin - 20 
                    
                    text_box_y1 = max(0, text_y - text_height - 4)
                    text_box_y2 = min(orig_h, text_y)
                    
                    cv2.rectangle(img, (xmin, text_box_y1), (xmin + text_width, text_box_y2), color, -1)
                    cv2.putText(img, label, (xmin, text_y - 2), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            # --- Save Image ---
            out_path = os.path.join(output_dir, f"val_comp_{img_name}")
            cv2.imwrite(out_path, img)

    print(f"[Visualizer] ✅ Visualization complete. {len(indices_to_visualize)} annotated images saved to: {output_dir}")

# ============================================================
# MAIN EXECUTION BLOCK
# ============================================================
if __name__ == "__main__":
    
    # --- Config (Must match your pipeline config) ---
    MODEL_NAME = "resnet152"
    
    # NOTE: Ensure 'project_dir' matches the location of your 'weights' folder.
    config = {
        "epochs": 100,
        "batch": 16,
        "imgsz": 224,
        # Based on your traceback, using the path ./resnet101/train_detr
        "project_dir": "./runs_152/train_detr", 
        "train_csv": os.path.join(DATA_ROOT, "train_processed.csv"),
        "val_csv": os.path.join(DATA_ROOT, "val_processed.csv"),
        "run_name": MODEL_NAME
    }
    
    # Run the visualization function
    visualize_validation_comparison(config, num_samples=100)
    print("\n✅ Validation Visualization script finished.")