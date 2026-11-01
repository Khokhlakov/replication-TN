"""Per-bin, per-channel normalizer fit on training rows only.

Trees do not use a normalized X_aux. ``signed_log1p`` is available for any later
neural use of the auxiliary block.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np


def signed_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def count_large(x: np.ndarray, threshold: float = 1e6) -> int:
    return int(np.sum(np.abs(x) > threshold))


def _row_hash(shape, row_idx: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(row_idx, dtype=np.int64).tobytes())
    return digest.hexdigest()


class Normalizer:
    def __init__(self, legacy_std0: bool = False):
        self.legacy_std0 = bool(legacy_std0)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.train_rows_: np.ndarray | None = None
        self.input_hash_: str | None = None

    def fit(self, x_train: np.ndarray, row_idx: np.ndarray | None = None) -> "Normalizer":
        x_train = np.asarray(x_train, dtype=np.float64)
        if x_train.ndim != 3:
            raise ValueError(f"expected (N, bins, channels), got {x_train.shape}")
        if row_idx is None:
            row_idx = np.arange(x_train.shape[0], dtype=np.int64)
        else:
            row_idx = np.asarray(row_idx, dtype=np.int64)
            if len(row_idx) != x_train.shape[0]:
                raise ValueError("row_idx length does not match the training matrix")
        self.mean_ = x_train.mean(axis=0)
        self.std_ = x_train.std(axis=0)
        self.train_rows_ = row_idx.copy()
        self.input_hash_ = _row_hash(x_train.shape[1:], row_idx)
        return self

    def transform(self, x: np.ndarray, legacy_std0: bool | None = None) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer.fit must be called before transform")
        flag = self.legacy_std0 if legacy_std0 is None else bool(legacy_std0)
        x = np.asarray(x, dtype=np.float64)
        std = self.std_
        scale = np.zeros_like(std)
        np.divide(1.0, std, out=scale, where=std != 0)
        out = (x - self.mean_) * scale
        if flag:
            dead = std == 0
            out[..., dead] = x[..., dead]
        return out.astype(np.float32, copy=False)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Normalizer":
        obj = joblib.load(path)
        if not isinstance(obj, Normalizer):
            raise TypeError(f"{path} is not a Normalizer")
        return obj
