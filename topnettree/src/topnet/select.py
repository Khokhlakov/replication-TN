"""Rank the 1536 CNN features with a gradient-boosted tree and keep the top ones."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from topnet.gbdt import make_gbdt


def rank_cnn_features(features: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> np.ndarray:
    """Fit on the provided rows only. Returns a permutation, most important first."""
    params = dict(cfg.get("select") or {})
    n_estimators = params.get("n_estimators")
    gbdt_cfg = {"gbdt": dict(cfg["gbdt"])}
    if n_estimators is not None:
        gbdt_cfg["gbdt"]["n_estimators"] = int(n_estimators)
    model = make_gbdt(gbdt_cfg, seed)
    model.fit(features, y)
    order = np.argsort(-model.feature_importances_, kind="mergesort")
    return order.astype(np.int64, copy=False)


def take_top(order: np.ndarray, n_keep: int) -> np.ndarray:
    return np.asarray(order[:n_keep], dtype=np.int64)


def bin_profile(idx: np.ndarray, k: int = 300, n_bins: int = 48) -> np.ndarray:
    used = np.asarray(idx[:k], dtype=np.int64)
    return np.bincount(used // 32, minlength=n_bins)


def save_bin_profile_plot(reference: np.ndarray, candidate: np.ndarray, path: str | Path) -> Path:
    """Write a histogram comparison. Falls back to a CSV if matplotlib is absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = bin_profile(reference)
    cand = bin_profile(candidate) if candidate.ndim == 1 and candidate.shape[0] != 48 else np.asarray(candidate)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3))
        bins = np.arange(48)
        ax.bar(bins - 0.2, ref, width=0.4, label="importance_idx.npy")
        ax.bar(bins + 0.2, cand, width=0.4, label="this run")
        ax.set_xlabel("filtration bin")
        ax.set_ylabel("count in top 300")
        ax.legend()
        fig.tight_layout()
        png = path if path.suffix == ".png" else path.with_suffix(".png")
        fig.savefig(png)
        plt.close(fig)
        return png
    except Exception:
        csv = path if path.suffix == ".csv" else path.with_suffix(".csv")
        stacked = np.column_stack([np.arange(48), ref, cand])
        np.savetxt(csv, stacked, delimiter=",", header="bin,reference,candidate", comments="")
        return csv
