"""
DETR Model for spacecraft detection (with Loss Curves)
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import SpacecraftDataset, TestSpacecraftDataset, DATA_ROOT, NUM_CLASSES, CLASS_NAMES, detr_collate_fn
import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from detr_model.detr import SetCriterion, PostProcess
from detr_model.__init__ import build_model
from detr_model.matcher import HungarianMatcher
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import nested_tensor_from_tensor_list

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# UTILITY FUNCTION
# ============================================================
def plot_losses(train_losses, val_losses, save_path):
    """Plots and saves the training and validation loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', linestyle='--')
    plt.plot(val_losses, label='Validation Loss', marker='o', linestyle='-')
    plt.title('DETR Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot to: {save_path}") 


# ============================================================
# ARGUMENTS/CONFIG UTILITY (DETR's build_model)
# ============================================================

class Args:
    """Simple class to hold required DETR configuration arguments."""
    def __init__(self, config):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.dataset_file = 'custom' 
        self.frozen_weights = None
        self.num_classes = config.get("num_classes", NUM_CLASSES + 1) 
        
        # --- Model Architecture ---
        self.backbone = 'resnet50' 
        self.dilation = True
        self.position_embedding = 'sine'
        self.masks = config.get("masks", False) 
        self.num_queries = 100 
        self.aux_loss = True
        self.lr_backbone = 1e-5 
        
        # --- Transformer parameters ---
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 8
        self.dec_layers = 8
        self.pre_norm = False
        self.clip_max_norm = 0.1 

        # --- Loss coefficients ---
        self.set_cost_class = 1 
        self.set_cost_bbox = 5  
        self.set_cost_giou = 2  
        self.bbox_loss_coef = 5 
        self.giou_loss_coef = 2 
        self.eos_coef = 0.3

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(config):
    set_seed(42)
    print("Training started...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build Model and Criterion
    detr_args = Args(config) 
    detr_args.num_classes = NUM_CLASSES + 1 
    model, criterion, postprocessors = build_model(detr_args)
    model.to(device)
    criterion.to(device)
    
    # Loss Tracking Initialization
    train_loss_history = []
    val_loss_history = []
    
    # Load datasets
    train_data = SpacecraftDataset(config["train_csv"], DATA_ROOT)
    val_data = SpacecraftDataset(config["val_csv"], DATA_ROOT)

    # DataLoader with custom collate_fn
    train_loader = DataLoader(train_data, batch_size=config["batch"], shuffle=True,
                              num_workers=8, pin_memory=True, collate_fn=detr_collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["batch"], shuffle=False,
                            num_workers=8, pin_memory=True, collate_fn=detr_collate_fn)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": detr_args.lr_backbone},
    ]
    opt = optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        opt, 
        mode='min', 
        factor=0.1, 
        patience=10
    )

    best_loss = float("inf") 

    best_iou = 0.0
    save_dir = os.path.join(config["project_dir"], "weights")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[DETR] Training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        # --- Training ---
        model.train()
        total_loss = 0
        for samples, targets in train_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            opt.zero_grad()
            outputs = model(samples)
            
            # Compute DETR loss
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            losses.backward()
            if detr_args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), detr_args.clip_max_norm)
                
            opt.step()
            total_loss += losses.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_total_loss = 0
        val_iou_sum = 0
        val_correct_classifications = 0
        matcher = criterion.matcher
        with torch.no_grad():
            for samples, targets in val_loader:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                val_total_loss += losses.item()

                # ===============================================
                # METRICS CALCULATION (IoU & Accuracy)
                # ===============================================
                indices = matcher(outputs, targets) # Get the optimal assignment 

                for i, (pred_idx, tgt_idx) in enumerate(indices):
                    # Only proceed if there is at least one Ground Truth object
                    if len(tgt_idx) == 0:
                        continue 
                    
                    # Ensure there is a matched prediction
                    if len(pred_idx) > 0 and len(tgt_idx) > 0:
                        # Classification Accuracy Check
                        # predicted class ID (excluding the no-object logit)
                        # take the top logit for the matched query (pred_idx[0])
                        pred_class = outputs['pred_logits'][i, pred_idx[0], :-1].argmax()
                        gt_class = targets[i]['labels'][tgt_idx[0]]
                        
                        if pred_class.item() == gt_class.item():
                            val_correct_classifications += 1

                        # IoU Calculation
                        # Normalized box coordinates for the matched query and target
                        pred_boxes_normalized = outputs['pred_boxes'][i, pred_idx]
                        tgt_boxes_normalized = targets[i]['boxes'][tgt_idx]

                        if pred_boxes_normalized.numel() > 0:
                            # generalized_box_iou returns IoU and GIoU
                            ious = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes_normalized),
                                                          box_cxcywh_to_xyxy(tgt_boxes_normalized))
                            
                            # taking the IoU values for the matched pairs (diagonal)
                            val_iou_sum += torch.diag(ious).sum().item()

        avg_val_loss = val_total_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        old_lrs = [g['lr'] for g in opt.param_groups]
        scheduler.step(avg_val_loss)
        new_lrs = [g['lr'] for g in opt.param_groups]

        if any(new_lr != old_lr for new_lr, old_lr in zip(new_lrs, old_lrs)):
            print(f"[LR] Epoch {epoch+1}: LR changed from {old_lrs} to {new_lrs}")
        
        num_val_samples = len(val_data)
        
        # Mean IoU (assumes 1 object per image)
        mean_iou = val_iou_sum / num_val_samples 
        
        # Detection Accuracy 
        detection_accuracy = (val_correct_classifications / num_val_samples) * 100

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | IoU: {mean_iou:.4f} | Acc: {detection_accuracy:.2f}%")

        # Save model
        if mean_iou > best_iou:
            best_iou = mean_iou
            model_path = os.path.join(save_dir, f"{config['run_name']}_best.pt")
            
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), model_path)
            print(f"Saved best model (IoU: {best_iou:.4f}) → {model_path}")

    # Plot and save losses after training
    loss_plot_path = os.path.join(config["project_dir"], f"{config['run_name']}_loss_history.png")
    plot_losses(train_loss_history, val_loss_history, loss_plot_path)

    print("Training complete.")


# ============================================================
# TESTING FUNCTION
# ============================================================
def test_model(config):
    """
    Loads trained DETR model, performs inference on test directory,
    and saves predictions (class + denormalized bounding box).
    """
    print(f"Starting inference on: {config['test_images_full_path']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model and Post-processor
    detr_args = Args(config)
    
    detr_args.num_classes = NUM_CLASSES + 1 
    model, _, postprocessors = build_model(detr_args)
    model.to(device)
    
    # Load weights
    model_path = os.path.join(config["project_dir"], "weights", f"{config['run_name']}_best.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model weights not found: {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from: {model_path}")

    # Load test data
    test_dir = config["test_images_full_path"]
    test_data = TestSpacecraftDataset(test_dir)
    test_loader = DataLoader(test_data, batch_size=config["batch"], shuffle=False,
                             num_workers=8, pin_memory=True)

    results = []

    # Inference loop
    with torch.no_grad():
        for imgs_tensor, img_names, H_tensor, W_tensor in test_loader:
            
            # Convert image batch to NestedTensor for DETR model input
            samples = [img.to(device) for img in imgs_tensor] 
            samples = nested_tensor_from_tensor_list(samples).to(device)
            
            outputs = model(samples)
            
            # Prepare original target sizes for post-processing/denormalization
            orig_target_sizes = torch.stack([
                torch.tensor([h, w], dtype=torch.float32, device=device) 
                for h, w in zip(H_tensor.tolist(), W_tensor.tolist())
            ])
            
            # PostProcess scales normalized boxes to pixel values (xyxy format)
            results_post = postprocessors['bbox'](outputs, orig_target_sizes)

            for i, res in enumerate(results_post):
                img_name = img_names[i]
                H, W = H_tensor[i].item(), W_tensor[i].item()
                
                # Get the top prediction
                scores, labels, boxes = res['scores'], res['labels'], res['boxes']
                
                if scores.numel() == 0:
                    class_name = "N/A"
                    bbox_str = "(0, 0, 0, 0)"
                else:
                    # Take the highest scoring detection
                    best_idx = torch.argmax(scores)
                    
                    # Boxes are already scaled to original image size (W, H) in xyxy format
                    xmin, ymin, xmax, ymax = boxes[best_idx].round().long().tolist()
                    class_id = labels[best_idx].item()
                    
                    # Clamp coordinates to bounds
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(W, xmax), min(H, ymax)

                    # Labels in DETR can be 0-9. Class 0 corresponds to CLASS_NAMES[0]
                    class_name = CLASS_NAMES[class_id]
                    bbox_str = f"({xmin}, {ymin}, {xmax}, {ymax})"

                results.append({
                    "Image name": img_name,
                    "Class": class_name,
                    "Bounding box": bbox_str
                })

    # 4. Save predictions
    output_path = os.path.join(config["project_dir"], f"{config['run_name']}_test_predictions.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Inference complete. Results saved to: {output_path}")


# ============================================================
# VISUALIZATION
# ============================================================
def visualize_predictions(config):
    print("Starting prediction visualization...")

    output_dir = os.path.join(config["project_dir"], "predictions_visualized")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(config["project_dir"], f"{config['run_name']}_test_predictions.csv")
    test_img_dir = config["test_images_full_path"]

    if not os.path.exists(csv_path):
        print(f"[ERROR] Prediction CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        img_name = row["Image name"]
        class_name = row["Class"]
        bbox_str = row["Bounding box"]

        try:
            xmin, ymin, xmax, ymax = [int(x.strip()) for x in bbox_str.strip("()").split(",")]
        except Exception:
            print(f"[WARN] Skipping malformed bbox: {bbox_str}")
            continue

        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not load image: {img_path}")
            continue

        # Draw bbox + label
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        label = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(img, (xmin, max(0, ymin - text_height - 4)), (xmin + text_width, ymin), color, -1)
        cv2.putText(img, label, (xmin, max(10, ymin - 2)), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, img)

    print(f"Visualization complete. Annotated images saved to: {output_dir}")