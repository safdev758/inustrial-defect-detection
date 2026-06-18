import torch
import torch.nn as nn
import torchvision


class ConvAutoencoder(nn.Module):
    """Trains on normal images only; reconstruction error = anomaly score."""
