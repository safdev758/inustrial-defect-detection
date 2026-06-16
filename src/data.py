from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def find_images(folder: Path):
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".bmp", ".tif"})


def defect_types(category_root: Path):
    test_root = category_root / "test"
    return sorted(d.name for d in test_root.iterdir() if d.is_dir() and d.name != "good")


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


