"""Gradient-boosted tree wrapper.

The sklearn backend follows the uploaded training code. An xgboost backend is
optional and is not numerically identical.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor


def gbdt_settings(cfg: dict, seed: int) -> tuple[str, dict]:
    raw = dict(cfg["gbdt"])
    backend = raw.pop("backend", "sklearn")
    raw["random_state"] = int(seed)
    return backend, raw


def make_gbdt(cfg: dict, seed: int):
    backend, params = gbdt_settings(cfg, seed)
    if backend == "sklearn":
        return GradientBoostingRegressor(**params)
    if backend == "xgboost":
        import xgboost

        return xgboost.XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            objective="reg:squarederror",
            random_state=int(seed),
            tree_method="hist",
            device="cuda",
        )
    raise ValueError(f"Unknown gbdt backend {backend}")


# Names that include H1/H2. Anything else is an ablation and is not TopNetTree.
TOPO_AUX = "topo_aux"
TOPO = "topo"
ABLATION_CNN_AUX = "cnn_aux"
ABLATION_CNN = "cnn"
ABLATION_AUX = "aux"
# Regenerated H1/H2 ablations. These names are not TopNetTree and are not the capsule vector.
ABLATION_H12 = "h12"
ABLATION_CNN_H12 = "cnn_h12"
ABLATION_CNN_AUX_H12 = "cnn_aux_h12"
H12_ORIGINAL = "h12_original"

FEATURE_SETS = (
    TOPO_AUX,
    TOPO,
    ABLATION_CNN_AUX,
    ABLATION_CNN,
    ABLATION_AUX,
    ABLATION_H12,
    ABLATION_CNN_H12,
    ABLATION_CNN_AUX_H12,
    H12_ORIGINAL,
)

_H12_SETS = (TOPO_AUX, TOPO, ABLATION_H12, ABLATION_CNN_H12, ABLATION_CNN_AUX_H12, H12_ORIGINAL)
_CNN_SETS = (TOPO_AUX, TOPO, ABLATION_CNN_AUX, ABLATION_CNN, ABLATION_CNN_H12, ABLATION_CNN_AUX_H12)
_AUX_SETS = (TOPO_AUX, ABLATION_CNN_AUX, ABLATION_AUX, ABLATION_CNN_AUX_H12)


def assemble_features(cnn_top: np.ndarray, x_aux, x_h12, feature_set: str):
    import numpy as np

    parts = []
    if feature_set in _CNN_SETS:
        parts.append(np.asarray(cnn_top))
    if feature_set in _H12_SETS:
        if x_h12 is None:
            raise FileNotFoundError(
                "feature set "
                f"{feature_set} needs an H1/H2 matrix. "
                "Refusing to rename an ablation to TopNetTree."
            )
        parts.append(np.asarray(x_h12))
    if feature_set in _AUX_SETS:
        if x_aux is None:
            raise FileNotFoundError(f"feature set {feature_set} needs X_aux")
        parts.append(np.asarray(x_aux))
    if not parts:
        raise ValueError(f"Unknown feature set {feature_set}")
    # cnn_aux_h12 order is cnn, then H1/H2, then aux, matching predict.py's concatenation.
    return np.concatenate(parts, axis=1)
