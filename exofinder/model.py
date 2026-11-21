import torch
from torch import nn


def encoder():
    return nn.Sequential(nn.Conv1d(2, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                         nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                         nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8), nn.Flatten())


class TransitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_encoder, self.local_encoder = encoder(), encoder()
        self.head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Dropout(.25), nn.Linear(64, 1))

    def forward(self, global_view, local_view):
        return self.head(torch.cat([self.global_encoder(global_view), self.local_encoder(local_view)], dim=1)).squeeze(1)


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
