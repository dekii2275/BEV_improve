"""Standalone KITTI evaluation script - runs bbox-only eval (no GPU IoU needed)"""
import sys
sys.path.insert(0, '/workspace/BEVHeight')

import os
import numpy as np
from evaluators.kitti_utils import kitti_common as kitti
from evaluators.kitti_utils.eval import eval_class, get_mAP_R40, image_box_overlap

def kitti_eval_bbox_only(pred_label_path, gt_label_path, current_classes=["Car", "Pedestrian", "Cyclist"]):
    """Evaluate using 2D bbox metric only (no GPU needed)"""
    pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
    gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
    print(f"Evaluating {len(pred_annos)} predictions vs {len(gt_annos)} ground truth")
    
    class_to_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Bus'}
    name_to_class = {v: n for n, v in class_to_name.items()}
    current_classes_int = [name_to_class[c] for c in current_classes]
    
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.7, 0.5, 0.5, 0.7, 0.5]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25],
                            [0.5, 0.25, 0.25, 0.5, 0.25]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    min_overlaps = min_overlaps[:, :, current_classes_int]
    
    difficultys = [0, 1, 2]
    difficulty_names = ['easy', 'moderate', 'hard']
    
    # Only run bbox metric (metric=0, no GPU needed)
    print("\n========== 2D BBox AP (R40) ==========")
    ret = eval_class(gt_annos, pred_annos, current_classes_int, difficultys, 0, min_overlaps)
    mAP_bbox = get_mAP_R40(ret['precision'])
    
    result = ""
    for j, curcls in enumerate(current_classes_int):
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            overlap_str = "strict" if i == 0 else "loose"
            result += f'\n{curcls_name} AP@{min_overlaps[i, 0, j]:.2f}, {min_overlaps[i, 1, j]:.2f}, {min_overlaps[i, 2, j]:.2f} ({overlap_str}):\n'
            result += f'  bbox AP: {mAP_bbox[j, 0, i]:.4f}, {mAP_bbox[j, 1, i]:.4f}, {mAP_bbox[j, 2, i]:.4f}\n'
    
    print(result)
    
    # Summary
    print("=" * 50)
    print("SUMMARY - 2D BBox AP (moderate difficulty):")
    print("=" * 50)
    for j, curcls in enumerate(current_classes_int):
        curcls_name = class_to_name[curcls]
        print(f"  {curcls_name}: {mAP_bbox[j, 1, 0]:.2f}% (strict) | {mAP_bbox[j, 1, 1]:.2f}% (loose)")
    
    if len(current_classes_int) > 1:
        mean_strict = np.mean([mAP_bbox[j, 1, 0] for j in range(len(current_classes_int))])
        mean_loose = np.mean([mAP_bbox[j, 1, 1] for j in range(len(current_classes_int))])
        print(f"  mAP:  {mean_strict:.2f}% (strict) | {mean_loose:.2f}% (loose)")

if __name__ == "__main__":
    pred_label_path = "outputs/data/"
    gt_label_path = "data/dair-v2x-i-kitti/training/label_2"
    kitti_eval_bbox_only(pred_label_path, gt_label_path)
