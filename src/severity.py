"""L10 — severity scoring: per-pixel AE error map + morphology → graded defect masks.

severity(image) = f(area of estimated defect mask, normalized error intensity inside it):
  grade LOW   < 0.15
  grade MEDIUM< 0.35
  grade HIGH  < 0.60
  grade CRITICAL >= 0.60

The error map is min-max normalized per image (max normal), the estimate mask is
the normalized map above 3x the good-pixel MAD (robust threshold, fitted on
train/good), then dilated (max-pool, k=5, it=2) + eroded once to kill speckle.
Against ground truth we report IoU and estimated-area MAE per defect type.
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import TrainNormalDataset, TestDataset

BANDS = ((0.15, "LOW"), (0.35, "MEDIUM"), (0.60, "HIGH"))
THRESH_MULT = 3.0


def _grade(intensity: float):
    for upper, name in BANDS:
        if intensity < upper:
            return name
    return "CRITICAL"


def robust_threshold(model, category_root, device, img_size=256, batch_size=32, log=print):
    """Threshold tau = good-mean + 3x good-MAD of per-pixel normalized error (fitted on train/good)."""
    loader = DataLoader(TrainNormalDataset(category_root, img_size), batch_size=batch_size, shuffle=False)
    errs = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            recon = model(imgs.to(device))
            abs_err = (recon - imgs.to(device)).abs().mean(dim=1)
            normed = abs_err / abs_err.amax(dim=(1, 2), keepdim=True)
            errs.append(normed.flatten().cpu())
    errs = torch.cat(errs).numpy()
    tau = float(errs.mean() + THRESH_MULT * np.abs(errs - errs.mean()).mean())
    log(f"[SEVERITY] pixel threshold tau={tau:.4f} (mean {errs.mean():.4f} + 3*MAD)")
    return tau


def _morphology(pred_mask: torch.Tensor, k=5, it=2):
    for _ in range(it):
        pred_mask = F.max_pool2d(pred_mask, k, 1, k // 2)
    return pred_mask


@torch.no_grad()
def evaluate_severity(model, category_root, device, img_size=256, batch_size=16, log=print):
    tau = robust_threshold(model, category_root, device, img_size=img_size, batch_size=batch_size, log=log)
    ds = TestDataset(category_root, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    per_type = defaultdict(list)
    area_mae, ious = [], []
    for imgs, labels, types, masks in loader:
        recon = model(imgs.to(device))
        abs_err = (recon - imgs.to(device)).abs().mean(dim=1)
        normed = abs_err / abs_err.amax(dim=(1, 2), keepdim=True)
        bin_mask = (normed > tau).float()
        bin_mask = _morphology(bin_mask)
        bin_mask = F.max_pool2d(-bin_mask, 3, 1, 1) * -1.0
        for i in range(imgs.shape[0]):
            est = bin_mask[i].cpu()
            normed_i = normed[i].cpu()
            area_pct = float(est.mean()) * 100.0
            intensity = float(normed_i[est.bool()].mean()) if est.any() else float(normed_i.max())
            grade = _grade(intensity)
            if labels[i] == 1:
                per_type[types[i]].append(grade)
                per_type[f"{types[i]}/area"].append(area_pct)
                gt = masks[i]
                inter = float((gt.bool() & est.bool()).sum())
                union = float((gt.bool() | est.bool()).sum())
                ious.append(inter / union if union else 1.0)
                area_mae.append(abs(float(gt.mean()) * 100.0 - area_pct))
            else:
                per_type["good/area"].append(area_pct)
    out = {"tau": round(tau, 4), "per_type": {}, "area_mae_pct": round(float(np.mean(area_mae)), 2), "mask_iou": round(float(np.mean(ious)), 3)}
    if per_type.get("good/area"):
        out["good_false_area_pct"] = round(float(np.mean(per_type["good/area"])), 2)
        log(f"[SEVERITY] good images: mean claimed-mask {out['good_false_area_pct']}% of image")
    for t in sorted(k for k in per_type if "/" not in str(k)):
        grades = per_type[t]
        dist = {g: grades.count(g) / len(grades) for g in ("LOW", "MEDIUM", "HIGH", "CRITICAL")}
        out["per_type"][t] = {
            "n": len(grades),
            "grade_distribution": {g: round(v, 3) for g, v in dist.items() if v > 0},
            "avg_estimated_area_pct": round(float(np.mean(per_type[f"{t}/area"])), 2),
        }
        log(f"[SEVERITY] {t}: n={len(grades)} grades={out['per_type'][t]['grade_distribution']}")
    log(f"[SEVERITY] area-MAE({out['area_mae_pct']}% of image) mask-IoU={out['mask_iou']}")
    return out
