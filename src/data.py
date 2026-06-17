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
                if mask_dir is not None:
                    stem = path.stem + "_mask"
                    matches = list(mask_dir.glob(stem + ".*"))
                    mask_path = matches[0] if matches else (mask_dir / (path.stem + ".png"))
                self.samples.append((path, 1 if type_folder.name != "good" else 0, type_folder.name, mask_path))
        self.t = make_transform(img_size)
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, defect_type, mask_path = self.samples[i]
        img = Image.open(path).convert("RGB")
        mask = torch.zeros((1, self.img_size, self.img_size))
        if mask_path is not None and mask_path.exists():
            m = Image.open(mask_path).convert("L")
            mask = transforms.functional.resize(transforms.functional.to_tensor(m), (self.img_size, self.img_size),
                                                transforms.InterpolationMode.NEAREST).round()
        return self.t(img), label, defect_type, mask


class SupervisedDataset(Dataset):
    """Classifier training set: train/good + test/good as negatives, chosen defect types as positives."""

    def __init__(self, category_root: Path, positive_types, img_size: int = 256):
        paths = []
        negatives = category_root / "train" / "good"
        paths += [(p, 0, "train_good") for p in find_images(negatives)]
        paths += [(p, 0, "test_good") for p in find_images(category_root / "test" / "good")]
        for t in positive_types:
            paths += [(p, 1, t) for p in find_images(category_root / "test" / t)]
        self.samples = paths
        self.t = make_transform(img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, defect_type = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.t(img), label, defect_type


def split_seen_unseen(category_root: Path):
    """First half of defect types (sorted) = 'seen' by the classifier, rest = novel."""
    dts = defect_types(category_root)
    half = (len(dts) + 1) // 2
    return dts[:half], dts[half:]
