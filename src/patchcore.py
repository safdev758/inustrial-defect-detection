"""PatchCore (Roth et al. 2022) — memory bank of normal patches + k-NN anomaly scoring.

Pipeline:
  1. Feature extractor: ImageNet-pretrained ResNet18, frozen; concat layer1+layer2 maps (192-d).
  2. Memory bank M: patch embeddings from train/good images, then greedy k-center coreset
     subsample (~10%) so nearest-neighbor scoring stays cheap.
  3. Scoring: b-score per test patch = sqrt(max(d2 - c, 0)) with c=1b (Roth's offset);
     image score = max patch score; pixel score = nearest-neighbor map upsampled.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from .data import TestDataset, TrainNormalDataset


def build_extractor(device):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net.to(device)


@torch.no_grad()
def _embed_batch(net, imgs, device):
    stem = net.maxpool(net.relu(net.bn1(net.conv1(imgs.to(device)))))
    l1 = net.layer1(stem)
    l2 = net.layer2(l1)
    l2 = F.interpolate(l2, size=l1.shape[2:], mode="bilinear")
    return torch.cat([l1, l2], dim=1)


@torch.no_grad()
def _collect_embeddings(net, loader, device):
    all_maps = []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        fmap = _embed_batch(net, imgs, device)
        B, C, H, W = fmap.shape
        all_maps.append(fmap.permute(0, 2, 3, 1).reshape(B * H * W, C))
    return torch.cat(all_maps).cpu()


def greedy_coreset(memory: torch.Tensor, ratio: float, device, log=print):
    """k-center greedy coreset (M = argamin max-dist selection)."""
    n_keep = max(1, int(memory.shape[0] * ratio))
    mem_gpu = memory.to(device)
    first = int(torch.randint(0, mem_gpu.shape[0], (1,)).item())
    selected = [first]
    dist = torch.cdist(mem_gpu[first:first + 1], mem_gpu).squeeze(0)
    for _ in range(1, n_keep):
        nxt = int(dist.argmax())
        selected.append(nxt)
        dist = torch.minimum(dist, torch.cdist(mem_gpu[nxt:nxt + 1], mem_gpu).squeeze(0))
    log(f"[PATCHCORE] coreset: {memory.shape[0]:,} -> {len(selected):,} patches")
    return memory[torch.tensor(selected)].to(device), selected


def train_patchcore(category_root: Path, img_size: int = 224, device: str = "cuda",
                    coreset_ratio: float = 0.1, batch_size: int = 16, log=print, memory_cap: int = 30_000):
    net = build_extractor(device)
    loader = DataLoader(TrainNormalDataset(category_root, img_size), batch_size=batch_size, shuffle=False)
    memory = _collect_embeddings(net, loader, device)
    if memory.shape[0] > memory_cap:
        rs = torch.Generator().manual_seed(42)
        keep = torch.randperm(memory.shape[0], generator=rs)[:memory_cap]
        memory = memory[keep]
    bank, _ = greedy_coreset(memory, coreset_ratio, device, log=log)
    return {"net": net, "bank": bank}


@torch.no_grad()
def evaluate_patchcore(model, category_root: Path, img_size: int = 224, device: str = "cuda",
                       batch_size: int = 16, log=print):
    from sklearn.metrics import roc_auc_score

    net, bank = model["net"], model["bank"]
    ds = TestDataset(category_root, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    image_scores, image_labels, dts_all, pixel_scores, pixel_labels = [], [], [], [], []
    for imgs, labels, dts, masks in loader:
        fmap = _embed_batch(net, imgs, device)
        B, C, H, W = fmap.shape
        patches = fmap.permute(0, 2, 3, 1).reshape(B * H * W, C)
        d2 = torch.cdist(patches, bank) ** 2
        b_scores = (d2.min(dim=1).values - 1.0).clamp_min(1e-8).sqrt()
        patch_map = b_scores.view(B, H, W).cpu().unsqueeze(1)
        up = F.interpolate(patch_map, size=imgs.shape[2], mode="bilinear").squeeze(1)
        for i in range(B):
            image_scores.append(float(b_scores[i * W * H:(i + 1) * W * H].max()))
            image_labels.append(int(labels[i]))
            dts_all.append(dts[i])
            pixel_scores.append(up[i].flatten())
            pixel_labels.append(masks[i].flatten() > 0)
    image_auroc = roc_auc_score(image_labels, image_scores)
    flat_p = np.concatenate([p.numpy() for p in pixel_scores])
    flat_l = np.concatenate([l.numpy().astype(int) for l in pixel_labels])
    pixel_auroc = roc_auc_score(flat_l, flat_p) if flat_l.any() else None
    log(f"[PATCHCORE-EVAL] image AUROC={image_auroc:.4f} "
        f"pixel AUROC={'n/a' if pixel_auroc is None else round(pixel_auroc, 4)}")
    def _safe_auroc(labels_, scores_):
        if len(set(labels_)) < 2:
            return None
        return round(roc_auc_score(labels_, scores_), 4)
    good_scores = [s for s, d in zip(image_scores, dts_all) if d == "good"]
    per_type = {}
    for dt in sorted(set(dts_all) - {"good"}):
        dt_scores = [s for s, d in zip(image_scores, dts_all) if d == dt]
        per_type[dt] = _safe_auroc([0] * len(good_scores) + [1] * len(dt_scores), good_scores + dt_scores)
        log(f"[PATCHCORE-EVAL] {dt}: AUROC={per_type[dt]} n_defect={len(dt_scores)}")
    return {"image_auroc": round(image_auroc, 4),
            "pixel_auroc": None if pixel_auroc is None else round(pixel_auroc, 4),
            "per_type_image_auroc": per_type}
