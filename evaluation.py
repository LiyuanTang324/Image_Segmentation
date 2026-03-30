import torch
import numpy as np
from medpy.metric.binary import hd95 as medpy_hd95


def get_DSC(pred, gt, threshold=0.5):
    """Dice Similarity Coefficient (per-sample, on GPU tensors)."""
    p = (pred > threshold).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum()
    return (2.0 * inter + 1e-8) / (p.sum() + g.sum() + 1e-8)


def get_IoU(pred, gt, threshold=0.5):
    """Intersection over Union / Jaccard Index (per-sample)."""
    p = (pred > threshold).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return (inter + 1e-8) / (union + 1e-8)


def get_HD95(pred, gt, threshold=0.5):
    """95th-percentile Hausdorff Distance (per-sample, on CPU/numpy).

    When one side is empty but the other is not, returns the image diagonal
    as a finite upper-bound so that downstream averaging stays meaningful.
    """
    pred_np = (pred > threshold).squeeze().cpu().numpy().astype(np.bool_)
    gt_np = (gt > 0.5).squeeze().cpu().numpy().astype(np.bool_)

    if pred_np.sum() == 0 and gt_np.sum() == 0:
        return 0.0
    if pred_np.sum() == 0 or gt_np.sum() == 0:
        h, w = pred_np.shape[-2:]
        return float(np.sqrt(h ** 2 + w ** 2))

    try:
        return float(medpy_hd95(pred_np, gt_np))
    except Exception:
        h, w = pred_np.shape[-2:]
        return float(np.sqrt(h ** 2 + w ** 2))
