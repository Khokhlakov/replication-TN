"""Fit and apply one CNN plus one GBDT on a stated feature set.

Models trained without X_h12 are ablations. Their names are the feature-set
names, not TopNetTree.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from topnet.cnn import TopNetCNN, extract_features, resolve_device, train_cnn
from topnet.features.manifest import LEGACY_NPY_MANIFEST, manifest_hash
from topnet.gbdt import assemble_features, make_gbdt
from topnet.normalize import Normalizer, count_large
from topnet.select import rank_cnn_features, take_top


def fit_pipeline(
    x_cnn: np.ndarray,
    y: np.ndarray,
    x_aux: np.ndarray | None,
    cfg: dict,
    seed: int,
    feature_set: str,
    row_idx: np.ndarray | None = None,
    x_h12: np.ndarray | None = None,
    device=None,
    manifest: dict | None = None,
) -> dict:
    if row_idx is None:
        row_idx = np.arange(len(y), dtype=np.int64)
    normalizer = Normalizer(legacy_std0=False)
    normalizer.fit(x_cnn, row_idx=row_idx)
    x_norm = normalizer.transform(x_cnn)
    model = train_cnn(x_norm, y, cfg, seed=seed, device=device)
    feats = extract_features(model, x_norm, device=device or resolve_device(cfg.get("device", "auto")))
    order = rank_cnn_features(feats, y, cfg, seed=seed + 1)
    n_keep = int(cfg["select"]["n_keep"])
    chosen = take_top(order, n_keep)
    matrix = assemble_features(feats[:, chosen], x_aux, x_h12, feature_set)
    gbdt = make_gbdt(cfg, seed + 2)
    gbdt.fit(matrix, y)
    aux_large = count_large(x_aux) if x_aux is not None else 0
    used_manifest = manifest if manifest is not None else LEGACY_NPY_MANIFEST
    return {
        "normalizer": normalizer,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cnn_kwargs": {
            "in_ch": int(cfg["cnn"]["in_ch"]),
            "drop1": float(cfg["cnn"]["drop1"]),
            "drop2": float(cfg["cnn"]["drop2"]),
            "drop3": float(cfg["cnn"]["drop3"]),
        },
        "importance_order": order,
        "chosen_idx": chosen,
        "gbdt": gbdt,
        "feature_set": feature_set,
        "row_idx": np.asarray(row_idx, dtype=np.int64),
        "aux_values_above_1e6": aux_large,
        "manifest": used_manifest,
        "manifest_hash": manifest_hash(used_manifest),
        "includes_h12": x_h12 is not None and feature_set in ("topo_aux", "topo"),
        "label": "no-H12" if feature_set in ("cnn_aux", "cnn", "aux") else "with-H12",
    }


def predict_pipeline(bundle: dict, x_cnn: np.ndarray, x_aux=None, x_h12=None, device=None) -> np.ndarray:
    normalizer: Normalizer = bundle["normalizer"]
    x_norm = normalizer.transform(x_cnn)
    model = TopNetCNN(**bundle["cnn_kwargs"])
    model.load_state_dict(bundle["state_dict"])
    dev = device or torch.device("cpu")
    model.to(dev)
    feats = extract_features(model, x_norm, device=dev)
    chosen = bundle["chosen_idx"]
    matrix = assemble_features(feats[:, chosen], x_aux, x_h12, bundle["feature_set"])
    return np.asarray(bundle["gbdt"].predict(matrix), dtype=np.float64)


def save_bundle(bundle: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if "manifest_hash" not in bundle or "gbdt" not in bundle:
        raise ValueError(f"{path} is not a TopNet artifact bundle")
    return bundle


def assert_manifest(bundle: dict, manifest: dict) -> None:
    got = manifest_hash(manifest)
    if got != bundle["manifest_hash"]:
        raise RuntimeError(
            "Feature manifest hash does not match the bundle "
            f"(bundle {bundle['manifest_hash']}, features {got}). Refusing to score."
        )
