import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .data import SupervisedDataset, TestDataset, split_seen_unseen
from .models import build_classifier


def train_classifier(category_root, seen_types, epochs=8, img_size=256, batch_size=32, lr=1e-4, device="cuda", log=print):
    train_ds = SupervisedDataset(category_root, seen_types, img_size=img_size)
    n_pos = sum(l for _, l, _ in train_ds.samples)
    n_neg = len(train_ds) - n_pos
    weight = torch.tensor([n_neg / max(1, n_neg), n_pos / max(1, n_pos)], device=device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model = build_classifier(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, running = 0, 0, 0.0
        for batch, labels, _ in loader:
            batch, labels = batch.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(batch)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * batch.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += batch.size(0)
        log(f"[CLF] epoch {epoch}/{epochs} loss={running / total:.4f} acc={correct / total:.3f}")
    return model


def evaluate_classifier(model, category_root, novel_types, img_size=256, device="cuda", log=print):
    test = TestDataset(category_root, img_size=img_size)
    loader = DataLoader(test, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    scores, labels, types = [], [], []
