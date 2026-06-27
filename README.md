# 1. Industrial Defect Detection & Quality Control

**Priority in portfolio:** #2 (build first after intro) — CNN, anomaly detection, segmentation
**Difficulty:** 8/10 · **Domain:** CNN (Computer Vision)

An AI quality-control system for manufacturing.

## What it does

- Receives images of manufactured products.
- Detects scratches, dents, cracks and other defects.
- Highlights defective regions using segmentation masks.
- Assigns a defect severity score.
- Produces an inspection report.
- Tracks defect frequency over time.

## Dataset

- **MVTec AD:** 5,000+ images, 15 categories, normal and defective samples.
- Alternative: KolektorSDD2.
- Link: https://www.mvtec.com/company/research/datasets/mvtec-ad
- Check MVTec's non-commercial research license before commercial use.

## Models

- Custom CNN from scratch
- ResNet
- Autoencoder for anomaly detection
- U-Net for defect segmentation

## What makes it impressive

Don't just classify defective vs. normal. Train an **anomaly detector using only
normal samples**, then evaluate whether it detects previously unseen defects.

> See the master document `../../ai-projects-portfolio.md` for the shared
> system architecture, the 9 mandatory ML-engineering requirements, and MLflow
> experiment tracking guidance.

---

### How to Run Your Own Training on Kaggle

#### Option 1: Standard Kaggle CLI

1. Inside your project folder, create `kernel-metadata.json`:

    ```json
    {
      "id": "khentitsafouaneamine/defect-detection-train",
      "title": "defect detection train",
      "code_file": "train.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": true,
      "enable_gpu": true,
      "enable_internet": true
    }
    ```

2. Push and launch training:

    ```bash
    kaggle kernels push -p .
    ```

3. Check status:

    ```bash
    kaggle kernels status khentitsafouaneamine/defect-detection-train
    ```

4. Download trained checkpoints / outputs:

    ```bash
    kaggle kernels output khentitsafouaneamine/defect-detection-train -p ./output
    ```

#### Option 2: The Multi-File Project Bundler (`kaggle_runner.py`)

Because Kaggle's API normally only uploads a single script, `kaggle_runner.py`
automatically packages whole project directories (`src/`, `models/`,
`requirements.txt`) and launches them:

```bash
# Bundle your whole project, enable GPU, watch progress, and auto-download results:
python3 kaggle_runner.py bundle --src ./01-industrial-defect-detection \
    --entry train.py --slug defect-detection-train --watch
```

> Move this folder's contents into `projects/01-industrial-defect-detection/`
> and run the bundle command from `projects/` (the parent), since the runner
> walks the source directory you point it at.

---

### Production serving (L10–L12): API + Docker + MLflow

The training stack and the serving stack are split on purpose — the API serves
`checkpoints/*.pth` (Kaggle artifacts, copied to `checkpoints/`) and
`deployments/app_metrics.json` (the Youden threshold from the run).

```bash
# 1. Quick sanity check with pytest (needs local torch):
python3 -m pytest tests/test_api.py -q

# 2. Serve locally without Docker:
cp output/train/checkpoints/*.pth checkpoints/   # pull from Kaggle first
uvicorn src.api:app --port 8000                  # then POST /inspect

# 3. Serve in Docker (CPU-only torch, ~1.5GB image):
docker build -t defect-inspection-api:1.0 .
docker run -d --name defect-api -p 8000:8000 defect-inspection-api:1.0
curl localhost:8000/health
curl -X POST localhost:8000/inspect -F "file=@image.png"

# 4. Log a pulled training run to MLflow:
python3 src/mlflow_log.py --project defect-detection \
    --kernel khentitsafouaneamine/defect-detection-train \
    --metrics output/train/output/metrics.json
```

Endpoints:
- `GET /health` → category + anomaly threshold
- `POST /inspect` (`multipart image`) → `decision` (ANOMALY/OK), `anomaly_score`,
  `threshold`, `margin`, plus classifier verdict restricted to seen defect types
  (`crack`, `cut`) or marker `unseen_type` for anything else.

### L13 — PatchCore memory bank (`src/patchcore.py`)

Frozen ImageNet ResNet18 feature extractor (layer1+layer2 concat, 192-d patch
embeddings) → memory bank of train/good patches → greedy k-center coreset
(~10%) → b-score nearest-neighbor scoring (Roth et al. 2022). Run with
`--phase patchcore` (no training loop: the "training" is building the bank).

Kernel results on hazelnut (vs the autoencoder baseline of L8):

| model | image AUROC | pixel AUROC |
| --- | --- | --- |
| ConvAutoencoder (L8) | 0.9543 | 0.9416 |
| **PatchCore (L13)** | **0.9625** | **0.9827** |

Teaching punchline: no gradient descent at all — the defects are simply "far
from every normal patch", and the coreset makes the distance search cheap.

### L13b — PatchCore per-type breakdown (kernel v2)

`evaluate_patchcore` now reports per-defect-type image AUROC (each type scored
against the good samples only, `per_type_image_auroc`, log lines
`[PATCHCORE-EVAL] <type>`). Hazelnut v2 (`output/patchcore_per_type`):

| defect type | image AUROC | n_defect |
| --- | --- | --- |
| crack | 0.9417 | 18 |
| cut | 0.9382 | 17 |
| hole | 0.9847 | 18 |
| print | 0.9853 | 17 |

Reading: bulk anomalies (hole areas, print patches) sit far from every coreset
patch → AUROC ≈ 0.98+; thin/linear anomalies (crack, cut) touch few patches, so
