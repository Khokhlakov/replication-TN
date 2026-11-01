"""YAML config loading. Paths resolve against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_LABEL_COLUMNS = (
    "row_idx",
    "pdb",
    "chain",
    "wt",
    "resid",
    "mut",
    "ddg",
    "is_nonbinder",
    "complex_id",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_update(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if "extends" in cfg:
        parent = path.parent / cfg.pop("extends")
        cfg = _deep_update(load_config(parent), cfg)
    cfg["_config_path"] = str(path.resolve())
    cfg["_root"] = str(repo_root())
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    raw = Path(cfg["paths"][key])
    if not raw.is_absolute():
        raw = Path(cfg["_root"]) / raw
    return raw
