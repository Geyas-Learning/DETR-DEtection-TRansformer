# detr_model/__init__.py

import torch

# FIXED IMPORTS to satisfy your required build_model structure
from .detr import build, SetCriterion, PostProcess 
from .segmentation import DETRsegm, PostProcessSegm
from .matcher import build_matcher
from .backbone import build_backbone
from .transformer import build_transformer
from util.misc import is_main_process 


def build_model(args):
    return build(args)


# ============================================================
# BUILD DETR SEGMENTATION MODEL (Corrected)
# ============================================================

def build_segmentation_model(args):
    # 1. Device
    device = torch.device(args.device)

    # 2. DETR Core
    
    # REQUIRED PREVIOUS FIX: Temporarily set args.masks=False to get the core DETR model
    original_masks = args.masks
    args.masks = False 
    
    # This now returns the original DETR class object
    detr_core_model, _, _ = build(args) 

    # Restore args.masks setting for criterion building
    args.masks = original_masks 

    # 3. DETRsegm wrapper 
    model = DETRsegm(detr_core_model, freeze_detr=(args.frozen_weights is not None))

    # 4. Hungarian Matcher
    matcher = build_matcher(args)

    # 5. Criterion/Losses
    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_mask': args.mask_loss_coef,
        'loss_dice': args.dice_loss_coef,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                f'loss_ce_{i}': 0.5,
                f'loss_bbox_{i}': 0.25,
                f'loss_giou_{i}': 0.25,
            })
        weight_dict.update(aux_weight_dict)


    losses = ['labels', 'boxes', 'cardinality', 'masks']
    
    # 6. SetCriterion
    criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    # 7. Postprocessors (FIXED: 'segm' is the correct key)
    postprocessors = {'bbox': PostProcess(), 'segm': PostProcessSegm()}
    
    return model, criterion, postprocessors