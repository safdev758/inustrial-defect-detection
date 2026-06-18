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
        dec = []
        pairs = [(a, b) for a, b in zip(ch[::-1], ch[::-1][1:])][:-1] + [(ch[1], ch[1])]
        for a, b in pairs:
            dec += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Upsample(scale_factor=2, mode="nearest")]
        self.decoder = nn.Sequential(*dec, nn.Conv2d(ch[1], ch[0], 3, padding=1), nn.Sigmoid())

