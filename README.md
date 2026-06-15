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
