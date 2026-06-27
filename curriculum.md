# Project 1 — Industrial Defect Detection: Full Curriculum (library-based / production)

**Stack:** PyTorch + torchvision + scikit-learn + matplotlib (local: CPU; training: Kaggle GPU `defect-detection-train`)
**Data:** MVTec AD via Kaggle dataset `ipythonx/mvtec-ad`, mounted at `/kaggle/input/mvtec-ad`

---

# Learning Framework (every lesson)

1. **Understand** — first-principles explanation before touching the framework's API.
2. **Do** — you immediately run the real code in this repo (`smoke` → `train.py`).
3. **Generate** — open-ended production task solved without scaffolding (modify this repo or write a script).
4. **Remember Separately** — flashcards below, kept apart from exploration.
5. **Spacing** — cold recall gate opens every session; drills re-queued at D+2, D+7, D+21.

---

# The repo you will be studying (implementation map)

| File | Role | Lesson |
|---|---|---|
| `src/data.py` | MVTec loaders: `TrainNormalDataset` (train/good), `TestDataset` (+ masks), `SupervisedDataset`, `split_seen_unseen` | L1, L2 |
| `src/models.py` | `ConvAutoencoder` (custom nn.Module), `build_classifier` (torchvision ResNet18) | L3, L5 |
| `src/unsupervised.py` | AE training loop + anomaly evaluation (image/pixel AUROC, Youden threshold, per-type table, ROC plot) | L4, L6 |
| `src/supervised.py` | Weighted-CE classifier training + seen/novel evaluation | L7, L8 |
| `train.py` | Orchestrator: args → seed → AE → classifier → `output/metrics.json` | L1 |
| `kernel-metadata.json` + `kaggle_runner.py` | Kaggle push / status / watch / pull | L9 |

Smoking-test the pipeline locally any time (synthetic tree ships in `data/smoke/`):

```bash
python3 train.py --data-root data/smoke --epochs 2 --classifier-epochs 2 --img-size 64 --batch-size 4 --phase both
```

---

# Phase 0 — PyTorch foundations read off the real pipeline

## L1. Datasets, loaders, tensors
- **Understand:** a `Dataset` is just `__len__` + `__getitem__`; a `DataLoader` adds batching/shuffling/worker processes and collates samples into *batched tensors*. Why `num_workers=2` and `pin_memory=True` exist (GPU feed rate).
- **Do:** edit the local smoke command's `--img-size` to 64/photo results; read `src/data.py` and trace one `__getitem__` → PIL image → `ToTensor` → `(3,H,W)` float tensor in `[0,1]`.
- **Generate:** without scaffolding, write your own `Dataset` that returns `(tensor, label)` from any folder of images, and swap it into `train_classifier` for a dry run.

## L2. Transforms & augmentation thinking
- **Understand:** `transforms.Resize/ToTensor`; why the AE pipeline deliberately uses NO augmentation (defects are the signal — geometric warps would teach the model that anything is acceptable geometry... discuss).
- **Generate:** list which augmentations would be safe for normal-only AE training, and which would destroy its ability to flag defects. Defend your choices.

## L3. nn.Module + the autoencoder as the impression of "normal"
- **Understand:** encoder (4 stride-2 stages: 3→32→64→128→256 channels, 256→16px), bottleneck = compressed "vocabulary of normality"; decoder mirrors it. BatchNorm train vs eval role.
- **Do:** `%python` print the model `repr`, count params; run one forward on a smoke tensor.
- **Generate:** compute the compression ratio; then predict what happens to anomaly contrast if you widen the AE (capacity) and only if you shrink it. Which direction destroys the method?

## L4. Training loop mechanics + losses
- **Understand:** zero_grad / forward / loss / backward / step — who holds the graph, when it dies; Adam vs SGD intuition; MSE on reconstruction vs L1.
- **Do:** launch `python3 train.py --epochs 3 --img-size 128` locally on smoke; watch the loss trajectory printed per epoch.

# Phase 1 — The anomaly system (official MVTec protocol: train on good only)

## L5. Scoring & evaluation
- **Understand:** image score = max per-pixel abs error (USER: Why max and not mean?); pixel AUROC from pixel-wise masks; ROC, Youden-J threshold; per-defect-type AUROC table = the real diagnostic.
- **Do:** with your AE, run evaluation against the real test set (Kaggle) and read `output/metrics.json` + `roc_curve.png`.
- **Generate:** modify `evaluate_autoencoder` to also emit the `AUROC using mean-of-top-k%` error, and argue in 5 sentences which score is better for a factory gate.

## L6. Failure analysis
- **Generate:** for the weakest defect type in the table, write one paragraph + 1 fix idea; queue it into the next training run's config.

# Phase 2 — Supervised branch + the novel-defect experiment

## L7. Transfer learning
- **Understand:** ImageNet-pretrained ResNet18 head swap to 2 logits; why `nn.CrossEntropyLoss(weight=...)` when positives are ~25% of data; AdamW.
- **Do:** run the `both` phase (default CLF epochs=8) on Kaggle and compare classifier vs AE on the same test set.

## L8. Seen/novel defect types
