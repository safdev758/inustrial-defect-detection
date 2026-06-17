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
