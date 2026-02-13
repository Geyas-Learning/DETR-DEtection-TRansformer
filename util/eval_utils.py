# ============================================================
# COCO EVALUATION LOGIC (Needs pycocotools installed)
# ============================================================
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# Note: You need to ensure 'util.misc' and 'util.box_ops' are importable
from util.misc import all_gather, is_main_process, reduce_dict
from util.box_ops import box_cxcywh_to_xyxy 


def get_coco_api_from_dataset(dataset):
    """
    Retrieves the pycocotools.COCO object from the dataset.
    NOTE: This assumes SpacecraftDataset either is or wraps the CocoDetection class 
    and exposes the COCO object via the '.coco' attribute.
    """
    if hasattr(dataset, 'coco'):
        return dataset.coco
    print("[COCO Eval] ⚠️ Warning: COCO API object not found on dataset. mAP will fail or be simulated.")
    return None 


class CocoEvaluatorPlaceholder:
    def __init__(self, coco_gt, iou_types):
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_results = []
        self.img_ids = set()
        # Fallback stats if real evaluation fails or is skipped
        self.base_stats = [0.0, 0.0] 

    def update(self, results):
        for img_id, output in results.items():
            boxes_xyxy = output['boxes'].cpu()
            labels = output['labels'].cpu()
            scores = output['scores'].cpu()

            # Convert [x0, y0, x1, y1] (pixel) to [x, y, w, h] (pixel) for COCOeval
            x0, y0, x1, y1 = boxes_xyxy.unbind(1)
            boxes_xywh = torch.stack([x0, y0, x1 - x0, y1 - y0], dim=1)

            for i in range(len(boxes_xyxy)):
                res = {
                    'image_id': img_id,
                    # NOTE: Assuming labels are the correct category IDs used in your COCO JSON
                    'category_id': labels[i].item(), 
                    'bbox': boxes_xywh[i].tolist(),
                    'score': scores[i].item(),
                }
                self.coco_results.append(res)
            self.img_ids.add(img_id)

    def synchronize_between_processes(self):
        """Gather all results from all processes to the main process."""
        all_results = all_gather(self.coco_results)
        all_img_ids = all_gather(list(self.img_ids))
        
        # Aggregate the gathered results on all processes (but only main will use it)
        self.coco_results = [res for res_list in all_results for res in res_list]
        self.img_ids = set(id for id_list in all_img_ids for id in id_list)

    def accumulate(self):
        """Computes the COCO metrics on the main process."""
        self.coco_eval = {"bbox": type('obj', (object,), {'stats': self.base_stats})()}
        
        if is_main_process() and self.coco_results:
            try:
                coco_dt = self.coco_gt.loadRes(copy.deepcopy(self.coco_results))
                coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
                coco_eval.params.imgIds = list(self.img_ids) 
                coco_eval.evaluate()
                coco_eval.accumulate()
                self.coco_eval["bbox"] = coco_eval
            except Exception as e:
                print(f"[COCO Eval Error] Failed during pycocotools process: {e}")

    def summarize(self):
        """Prints and returns the summary stats."""
        stats = self.base_stats
        if is_main_process():
            if hasattr(self.coco_eval["bbox"], 'stats'):
                self.coco_eval["bbox"].summarize()
                stats = self.coco_eval["bbox"].stats
        return stats
    

def run_coco_evaluation(model, criterion, postprocessors, data_loader, device, detr_args):
    """
    Performs inference and computes COCO metrics (mAP) over the entire validation set.
    """
    model.eval()
    criterion.eval()
    
    # 1. Get COCO Ground Truth API
    base_ds = get_coco_api_from_dataset(data_loader.dataset)
    if base_ds is None:
        if is_main_process():
            print("[COCO Eval] 🛑 Cannot run real COCO evaluation. Returning simulated metrics.")
        return 0.5, 0.0, 0.0 # Fake loss, mAP, mAP50

    coco_evaluator = CocoEvaluatorPlaceholder(base_ds, iou_types=("bbox",))
    val_total_loss = torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        for imgs_tensor, targets in data_loader:
            # Use nested_tensor_from_tensor_list for the batch processing (from train_model logic)
            samples = nested_tensor_from_tensor_list([img.to(device) for img in imgs_tensor.tensors]).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(samples)
            
            # Loss calculation (necessary for the loss term)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            val_total_loss += losses.item()
            
            # Evaluation Step
            # targets must contain 'orig_size' and 'image_id' (provided by coco.py)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            
            # Post-process (normalized cxcywh -> pixel xyxy/filtered)
            results_post = postprocessors['bbox'](outputs, orig_target_sizes)
            
            # Format results for COCO evaluation
            res = {target['image_id'].item(): output for target, output in zip(targets, results_post)}
            coco_evaluator.update(res)

    # 2. Synchronize and Compute Metrics
    
    # Average loss across all GPUs
    avg_loss_on_process = val_total_loss / len(data_loader)
    loss_dict_to_reduce = {'val_loss': torch.tensor(avg_loss_on_process, device=device)}
    loss_dict_reduced = reduce_dict(loss_dict_to_reduce)
    avg_val_loss = loss_dict_reduced['val_loss'].item()
    
    # Gather results to master process and compute metrics
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
        
    # stats[0] is mAP@[0.5:0.95], stats[1] is mAP@0.50
    return avg_val_loss, stats[0].item(), stats[1].item()