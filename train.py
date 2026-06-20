import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import split_seen_unseen
from src.patchcore import evaluate_patchcore, train_patchcore
from src.supervised import evaluate_classifier, train_classifier
from src.unsupervised import evaluate_autoencoder, train_autoencoder


def parse_args():
    p = argparse.ArgumentParser(description="Industrial Defect Detection — MVTec AD")
    root = p.add_argument("--data-root", type=str, default=None, help="MVTec AD root (auto-detects /kaggle/input/*)")
    p.add_argument("--categories", nargs="+", default=["hazelnut"])
    p.add_argument("--phase", choices=["unsup", "both", "patchcore", "severity"], default="unsup")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--classifier-epochs", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def detect_data_root(arg_root):
    if arg_root:
        return Path(arg_root)
    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        print(f"[ENV] mounted inputs: {sorted(str(p) for p in input_dir.iterdir())}")
        for base in sorted(input_dir.iterdir()):
            for cand in [base, *sorted(base.iterdir())]:
                if not cand.is_dir():
                    continue
                for c in [cand, *sorted(cand.iterdir())]:
                    if c.is_dir() and (c / "hazelnut" / "train" / "good").exists():
                        return c
        seen = []
        for base in sorted(input_dir.rglob("*")):
            if len(base.parts) <= len(input_dir.parts) + 4:
                seen.append(str(base))
        print("input tree:", "\n".join(seen))
    raise SystemExit("No MVTec AD root found. Pass --data-root.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    data_root = detect_data_root(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] device={device} data_root={data_root} categories={args.categories} phase={args.phase}")

    output_dir = Path("output")
    ckpt_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    metrics = {"meta": vars(args) | {"data_root": str(data_root), "device": device}}
    for category in args.categories:
        print(f"\n{'=' * 60}\n🚂 category: {category}\n{'=' * 60}")
