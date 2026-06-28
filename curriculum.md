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
- **Understand:** `split_seen_unseen` trains the classifier on half the defect *types* and tests on never-seen ones — the exact "detects previously unseen defects" claim in a measurable form.
- **Generate:** write 5-sentence verdict: does supervised transfer learning generalize to novel defect types? Where does the AE win/lose? (Deliverable: the chart + the paragraph in your portfolio.)

# Phase 3 — Kaggle ops (library versions of the whole workflow)

## L9. Kernel + runner
- **Understand:** `kernel-metadata.json` schema; the runner embeds the whole project as a base64 zip (multi-file), mounts Kaggle datasets, runs the entry script, streams checkpoints to `/kaggle/working/output`.
- **Do:** smoke push → `status` → push full run → `watch` → `pull` artifacts (steps below in README Option 1/2).
- **Generate:** attach MLflow-free run tracking: extend `train.py`'s argparse to take `--epochs-by-category` dict, and make a second kernel slug that trains two categories sequentially.

# Phase 4 — Production hardening (build after L9, one at a time)
- **L10.** Segmentation branch: U-Net-style AE with per-pixel error + morphology → **severity score** (mask area × intensity bands).
- **L11.** FastAPI `/inspect` endpoint + pydantic response model (decision, margin, defect areas) + Dockerfile. ✅ **implemented — see README "Production serving"**: `src/api.py` (lifespan-loads AE + classifier, `/health` + `/inspect`), `tests/test_api.py` (3 passing), `Dockerfile` (CPU-only torch), MLflow script `src/mlflow_log.py` (28 metrics from the real run in experiment `defect-detection_training`).
  - Lesson learned (part of L11 study): the synthetic normal-like square got flagged **ANOMALY** by the real kernel — the AE treats any OOD input as anomalous. Your Generate task: design a calibration + drift-monitoring story for a factory where the lighting changes seasonally.
- **L12.** MLflow experiment tracking for every run config → the comparison table habit (same discipline as project 5).
- **L13.** PatchCore brief: memory bank of normal latent features as the industry go-to; implement a 50-line variant and compare to your AE in `metrics.json`. ✅ **implemented — `src/patchcore.py`** (frozen ResNet18 ImageNet extractor, layer1+2 concat, greedy k-center coreset 10%, b-score kNN), kernel `defect-detection-patchcore`: image 0.9625 / pixel 0.9827 vs AE's 0.9543/0.9416 — logged to MLflow experiment `defect-detection-patchcore_training`. C14–C15 below extend the L13 review.

---

# Flashcards (spaced review — never explore with these in front of you)

- C1 (L1): What contract must a `torch.utils.data.Dataset` implement?
- C2 (L1): Why is `pin_memory=True` irrelevant on a CPU-only run?
- C3 (L3): AE encoder/decoder channel plan in our model, in numbers (3→…→…→3)?
- C4 (L4): Name the 5 lines and the right *order* of a training step.
- C5 (L5): Difference between image-level and pixel-level AUROC — what does each answer?
- C6 (L5): Youden's J = ? (formula)
- C7 (L8): Why not train the supervised classifier on ALL defect types and report its test accuracy on the same ones?
- C8 (L9): Who writes `kernel-metadata.json` in Option 2 — you or the runner? Which fields?
- C9 (L13): What goes into the memory bank in a PatchCore approach?
- C11 (L11): In `src/api.py` lifespan — why load the AE once at startup instead of per-request? (3 answers)
- C12 (L11): What does the container flag when the score is BELOW threshold — and what classifier verdict does the API emit in that case?
- C13 (L11): The synthetic square was flagged ANOMALY — name the two monitoring signals (1 calibration, 1 drift) you would set up to distinguish bug from reality.
- C14 (L13): Why does PatchCore need *no gradient descent*? What is the only "training" step?
- C15 (L13): What does the greedy k-center coreset trade away (bank coverage) vs buy (latency), and how would you check the inflation of image AUROC if you raised the ratio 10%→100%?

# Spacing protocol
- Every session: 4 random cards, cold.
- D+2 after L3: redraw the AE architecture from memory.
- D+2 after L5: re-explain "why max-per-pixel" to the rubber duck.
- D+7 after L8: reprint the classifier+AE table from memory before re-reading `metrics.json`.
- D+21 after L9: re-push a full Kaggle run from scratch with no README.

# Final gate (portfolio checklist — the same 9 in the master doc)
EDA → baseline → implement → experiment → evaluate → deploy → containerize → monitor → document.
- **L10.** Segmentation branch: U-Net-style AE with per-pixel error + morphology → **severity score** (mask area × intensity bands). ✅ **implemented — `src/severity.py`**: robust `tau` from the good distribution, morphological cleanup, grade bands, ground-truth IoU/area-MAE probes, kernel `defect-detection-l10-severity` v2 (crack MEDIUM 100% / cut LOW 65% / hole MEDIUM 78% / print MEDIUM 94%), false-area probe `good_false_area_pct`=26.63% — logged to `defect-detection-severity_training`. Flashcards C16–C18 below.
- C16 (L10): Why is the error map normalized per-image by its MAX — and what bias does that introduce on defect-free frames? (answer: `good_false_area_pct`)
- C17 (L10): Tau = mean + 3×MAD — why MAD instead of std? What breaks if you used std here?
- C18 (L10): Why is the reported IoU (0.28) so low even though the grades look plausible — name the two error sources in the map itself.
