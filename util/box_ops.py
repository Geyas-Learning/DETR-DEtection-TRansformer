# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def complete_box_iou(boxes1, boxes2):
    """
    Complete IoU (CIoU) Loss implementation.
    The boxes should be in [x0, y0, x1, y1] format (XYXY).
    
    Returns L_CIoU (which is minimized).
    """
    # 1. Calculate IoU
    iou, union = box_iou(boxes1, boxes2)
    
    # Ensure all dimensions are broadcastable
    boxes1 = boxes1.unsqueeze(1) if boxes1.dim() == 2 else boxes1
    boxes2 = boxes2.unsqueeze(0) if boxes2.dim() == 2 else boxes2
    
    # 2. Distance term (p^2 / c^2)
    
    # Get center points (cx, cy)
    boxes1_cxcy = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    boxes2_cxcy = (boxes2[..., :2] + boxes2[..., 2:]) / 2
    
    # Distance between center points (p^2)
    center_dist_sq = torch.sum((boxes1_cxcy - boxes2_cxcy) ** 2, dim=-1)
    
    # Coordinates of the smallest enclosing box (C)
    C_xmin = torch.min(boxes1[..., 0], boxes2[..., 0])
    C_ymin = torch.min(boxes1[..., 1], boxes2[..., 1])
    C_xmax = torch.max(boxes1[..., 2], boxes2[..., 2])
    C_ymax = torch.max(boxes1[..., 3], boxes2[..., 3])
    
    # Diagonal of the smallest enclosing box squared (c^2)
    C_diag_sq = (C_xmax - C_xmin) ** 2 + (C_ymax - C_ymin) ** 2
    
    # Avoid division by zero
    C_diag_sq = torch.clamp(C_diag_sq, min=1e-6)
    
    # Distance ratio term (u)
    u = center_dist_sq / C_diag_sq

    # 3. Aspect Ratio Consistency Term (v)
    
    # Box widths and heights
    boxes1_w = boxes1[..., 2] - boxes1[..., 0]
    boxes1_h = boxes1[..., 3] - boxes1[..., 1]
    boxes2_w = boxes2[..., 2] - boxes2[..., 0]
    boxes2_h = boxes2[..., 3] - boxes2[..., 1]
    
    # Avoid log/tan issues with zero dimensions
    boxes1_w = torch.clamp(boxes1_w, min=1e-6)
    boxes1_h = torch.clamp(boxes1_h, min=1e-6)
    boxes2_w = torch.clamp(boxes2_w, min=1e-6)
    boxes2_h = torch.clamp(boxes2_h, min=1e-6)
    
    v = (4 / (np.pi**2)) * torch.pow(
        torch.atan(boxes1_w / boxes1_h) - torch.atan(boxes2_w / boxes2_h), 2
    )
    
    # 4. Alpha (Trade-off parameter)
    # alpha is defined based on the loss (1 - iou)
    alpha = v / (1 - iou + v + 1e-6)
    
    # 5. CIoU Loss: L_CIoU = 1 - IoU + u + alpha * v
    # Note: We return L_CIoU, which is the quantity to be minimized
    ciou_loss = 1 - iou + u + alpha * v
    
    return ciou_loss


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

