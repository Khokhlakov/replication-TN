"""TopNet CNN. Input is stored as (N, 48, 108) and fed channels-first.

``features`` flattens in Keras order: index = bin * 32 + filter.
"""

from __future__ import annotations

import math
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def keras_flatten(h: torch.Tensor) -> torch.Tensor:
    """(B, 32, 48) filter-major activation -> (B, 1536) bin-major Keras flatten."""
    return h.permute(0, 2, 1).reshape(h.shape[0], -1)


def _truncated_normal_(tensor: torch.Tensor, std: float) -> None:
    with torch.no_grad():
        tensor.normal_(0.0, std)
        mask = tensor.abs() > (2.0 * std)
        while bool(mask.any()):
            n = int(mask.sum().item())
            redraw = torch.normal(0.0, std, size=(n,), device=tensor.device, dtype=tensor.dtype)
            tensor.masked_scatter_(mask, redraw)
            mask = tensor.abs() > (2.0 * std)


def _init_conv(layer: nn.Conv1d, kind: str) -> None:
    fan_in = layer.weight.shape[1] * layer.weight.shape[2]
    with torch.no_grad():
        if kind == "he_normal":
            _truncated_normal_(layer.weight, math.sqrt(2.0 / fan_in))
        elif kind == "lecun_uniform":
            limit = math.sqrt(3.0 / fan_in)
            layer.weight.uniform_(-limit, limit)
        else:
            raise ValueError(kind)
        layer.bias.zero_()


def _init_linear(layer: nn.Linear) -> None:
    fan_in = layer.weight.shape[1]
    fan_out = layer.weight.shape[0]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        layer.weight.uniform_(-limit, limit)
        layer.bias.zero_()


class TopNetCNN(nn.Module):
    def __init__(self, in_ch: int = 108, drop1: float = 0.3, drop2: float = 0.3, drop3: float = 0.1):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, 16, 3, padding=1)
        self.c2 = nn.Conv1d(16, 16, 3, padding=1)
        self.c3 = nn.Conv1d(16, 32, 3, padding=1)
        self.c4 = nn.Conv1d(32, 32, 3, padding=1)
        self.d1 = nn.Dropout(drop1)
        self.d2 = nn.Dropout(drop2)
        self.d3 = nn.Dropout(drop3)
        self.fc = nn.Linear(48 * 32, 1)
        _init_conv(self.c1, "he_normal")
        _init_conv(self.c2, "lecun_uniform")
        _init_conv(self.c3, "lecun_uniform")
        _init_conv(self.c4, "lecun_uniform")
        _init_linear(self.fc)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x is (B, 108, 48). Returns the 1536-d flatten, before the last dropout."""
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = self.d1(h)
        h = F.relu(self.c3(h))
        h = F.relu(self.c4(h))
        h = self.d2(h)
        return keras_flatten(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.d3(self.features(x))).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str | None) -> torch.device:
    if name is None or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def device_report(device: torch.device) -> dict:
    info = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def train_cnn(
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    seed: int,
    device: torch.device | str | None = None,
) -> TopNetCNN:
    """Train on the rows of ``x`` and ``y``. The caller must already have sliced them."""
    cnn_cfg = cfg["cnn"]
    seed_everything(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if device is None:
        device = resolve_device(cfg.get("device", "auto"))
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    model = TopNetCNN(
        in_ch=int(cnn_cfg["in_ch"]),
        drop1=float(cnn_cfg["drop1"]),
        drop2=float(cnn_cfg["drop2"]),
        drop3=float(cnn_cfg["drop3"]),
    ).to(device)

    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    xt = xt.permute(0, 2, 1).to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1)).to(device)
    n = xt.shape[0]
    batch = int(cnn_cfg["batch_size"])
    epochs = int(cnn_cfg["epochs"])
    opt = torch.optim.Adam(model.parameters(), lr=float(cnn_cfg["lr"]), eps=float(cnn_cfg["adam_eps"]))
    generator = torch.Generator()
    generator.manual_seed(seed)

    model.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n, generator=generator).tolist()
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            xb = xt[idx]
            yb = yt[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
    return model


def extract_features(model: TopNetCNN, x: np.ndarray, device: torch.device | str | None = None, batch: int = 64) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    was_training = model.training
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(0, 2, 1)
    chunks = []
    with torch.no_grad():
        for start in range(0, xt.shape[0], batch):
            xb = xt[start : start + batch].to(device)
            chunks.append(model.features(xb).detach().cpu())
    model.train(was_training)
    return torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)
