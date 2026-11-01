"""Write a self-contained run directory: config, versions, predictions, metrics."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from topnet import __version__
from topnet.config import repo_root


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def start_run(name: str, cfg: dict, synthetic: bool = False) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "SYNTHETIC_" if synthetic else ""
    path = repo_root() / "runs" / f"{prefix}{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    public = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    (path / "config.yaml").write_text(yaml.safe_dump(public, sort_keys=False), encoding="utf-8")
    meta = {
        "name": name,
        "synthetic": synthetic,
        "created_utc": stamp,
        "package": __version__,
        "git": _git_hash(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": platform.platform(),
        "seed": cfg.get("seed"),
        "seeds": cfg.get("cv", {}).get("seeds"),
    }
    (path / "environment.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def write_outputs(path: Path, predictions: pd.DataFrame, metrics: dict) -> None:
    csv_path = path / "predictions.csv"
    predictions.to_csv(csv_path, index=False)
    try:
        predictions.to_parquet(path / "predictions.parquet", index=False)
    except Exception:
        pass
    (path / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
