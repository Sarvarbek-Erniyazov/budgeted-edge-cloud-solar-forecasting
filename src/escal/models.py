"""Edge and cloud tiers. Both predict the benchmark kt target for all horizons at once."""
from __future__ import annotations

import torch
from torch import nn


def mlp(sizes: list[int]) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


class EdgeNet(nn.Module):
    """Plain Linear+ReLU MLP: quantises to int8 without special layers."""

    def __init__(self, n_in: int, n_out: int, hidden: list[int]):
        super().__init__()
        self.body = mlp([n_in] + hidden)
        self.head = nn.Linear(hidden[-1], n_out)

    def forward(self, x, tiles=None):
        return self.head(self.body(x))


class CloudNet(nn.Module):
    """Fusion: CNN over the satellite tile stack + MLP over ground, NAM and tile metadata."""

    def __init__(self, n_in: int, n_out: int, n_chan: int, hidden: list[int], cnn: list[int], fusion: int):
        super().__init__()
        convs, c = [], n_chan
        for k in cnn:
            convs += [nn.Conv2d(c, k, 3, padding=1), nn.ReLU()]
            c = k
        self.cnn = nn.Sequential(*convs)
        self.tab = mlp([n_in] + hidden)
        self.head = nn.Sequential(nn.Linear(hidden[-1] + cnn[-1], fusion), nn.ReLU(), nn.Linear(fusion, n_out))

    def forward(self, x, tiles):
        # spatial mean instead of AdaptiveAvgPool2d: same value, deterministic backward on CUDA
        return self.head(torch.cat([self.tab(x), self.cnn(tiles).mean(dim=(2, 3))], dim=1))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
