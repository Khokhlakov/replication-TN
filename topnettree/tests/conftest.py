from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg() -> dict:
    return {
        "seed": 0,
        "device": "cpu",
        "cnn": {
            "in_ch": 108,
            "drop1": 0.0,
            "drop2": 0.0,
            "drop3": 0.0,
            "lr": 1e-4,
            "adam_eps": 1e-7,
            "batch_size": 8,
            "epochs": 1,
            "loss": "mse",
        },
        "select": {"n_keep": 8, "n_estimators": 3},
        "gbdt": {
            "n_estimators": 3,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_samples_split": 2,
            "subsample": 1.0,
            "max_features": "sqrt",
            "loss": "squared_error",
            "backend": "sklearn",
        },
        "cv": {
            "mode": "nested",
            "splitter": "kfold10",
            "n_splits": 3,
            "seeds": [0],
            "feature_set": "cnn_aux",
        },
        "tune": {
            "max_depth": [2],
            "subsample": [1.0],
            "n_estimators": [2],
            "inner_splits": 3,
        },
    }


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"topnet_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cfg():
    return tiny_cfg()


@pytest.fixture
def script_loader():
    return load_script
