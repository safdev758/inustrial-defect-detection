import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from .data import TestDataset, TrainNormalDataset
from .models import ConvAutoencoder


def train_autoencoder(category_root, epochs=30, img_size=256, batch_size=32, lr=2e-4, device="cuda", log=print):
    dataset = TrainNormalDataset(category_root, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * batch.size(0)
        recon_ok = (batch - recon).abs().mean().item()
        log(f"[AE] epoch {epoch}/{epochs} mse={total / len(dataset):.5f} mean-abs-err={recon_ok:.5f}")
    return model


def evaluate_autoencoder(model, category_root, img_size=256, device="cuda", output_dir=None, log=print):
    test = TestDataset(category_root, img_size=img_size)
    loader = DataLoader(test, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    img_err, img_label, pixel_err, pixel_label, per_type = [], [], [], [], {}
    good_errs = []
    with torch.no_grad():
        for batch, label, types, masks in loader:
            batch = batch.to(device)
            recon = model(batch)
            abs_err = (recon - batch).abs().mean(dim=1)  # HxW per pixel
            img_err += abs_err.amax(dim=(1, 2)).cpu().tolist()
            img_label += label.tolist()
            pixel_err.append(abs_err.cpu().numpy().ravel())
            pixel_label.append(masks.squeeze(1).numpy().ravel())
            for t, e, l in zip(types, abs_err.amax(dim=(1, 2)).cpu().tolist(), label.tolist()):
                if l == 0:
                    good_errs.append(e)
                else:
                    per_type.setdefault(t, []).append(e)
    pixel_pred = np.concatenate(pixel_err)
    pixel_true = np.concatenate(pixel_label)
    pixel_auroc = roc_auc_score(pixel_true, pixel_pred)
    image_auroc = roc_auc_score(img_label, img_err)
    fpr, tpr, thr = roc_curve(img_label, img_err)
    best_i = int(np.argmax(tpr - fpr))
    threshold = thr[best_i]
    confusion = evaluate_threshold(img_err, img_label, threshold)
    result = {
        "image_auroc": round(image_auroc, 4),
        "pixel_auroc": round(pixel_auroc, 4),
        "best_threshold": round(float(threshold), 5),
        "best_confusion_matrix": confusion,
        "per_defect_type": {},
    }
    for t, e_list in per_type.items():
        e = np.array(e_list)
        scores = np.concatenate([e, np.array(good_errs)])
        labels = np.concatenate([np.ones(len(e)), np.zeros(len(good_errs))])
