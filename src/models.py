import torch
import torch.nn as nn
import torchvision


class ConvAutoencoder(nn.Module):
    """Trains on normal images only; reconstruction error = anomaly score."""

    def __init__(self):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        enc = []
        for a, b in zip(ch, ch[1:]):
            enc += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.encoder = nn.Sequential(*enc)
