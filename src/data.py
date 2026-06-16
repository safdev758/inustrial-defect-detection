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


class TrainNormalDataset(Dataset):
    """train/good only — the official MVTec training partition."""

    def __init__(self, category_root: Path, img_size: int = 256):
        self.paths = find_images(category_root / "train" / "good")
        self.t = make_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.t(img)


class TestDataset(Dataset):
    """test/good + test/<defect_type>, with mask lookup for pixel metrics."""

    def __init__(self, category_root: Path, img_size: int = 256):
        self.samples = []  # (path, label, defect_type, mask_path or None)
        test_root = category_root / "test"
        gt_root = category_root / "ground_truth"
        for type_folder in sorted(p for p in test_root.iterdir() if p.is_dir()):
            mask_dir = gt_root / type_folder.name if type_folder.name != "good" else None
            for path in find_images(type_folder):
                mask_path = None
