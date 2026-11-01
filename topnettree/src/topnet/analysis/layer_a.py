"""Layer A statistics on a mutation-by-complex BFE-change matrix.

The comparison is strict. Thresholds are the published list, not an arange.
The exponent uses kT = 1. kT = 0.593 is a labeled sensitivity option only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

THRESH = [0, -0.3, -0.6, -0.9, -1.2, -1.5, -1.8, -2.1, -2.4, -2.7, -3.0]

BA1_MUTATIONS = [
    "G339D",
    "S371L",
    "S373P",
    "S375F",
    "K417N",
    "N440K",
    "G446S",
    "S477N",
    "T478K",
    "E484A",
    "Q493R",
    "G496S",
    "Q498R",
    "N501Y",
    "Y505H",
]


def disrupted_counts(p: np.ndarray, thresholds=THRESH) -> list[int]:
    """A complex is disrupted at t when its most negative single-mutation change is < t."""
    lowest = np.nanmin(p, axis=0)
    return [int((lowest < t).sum()) for t in thresholds]


def fold_reduction(p: np.ndarray, kT: float = 1.0) -> float:
    """exp(-mean over complexes of the sum over mutations of min(P, 0))."""
    clipped = np.clip(p, None, 0.0)
    return float(np.exp(-clipped.sum(axis=0).mean() / kT))


def infectivity_ratio(sum_a: float, sum_b: float) -> float:
    """exp(sum_a - sum_b). No RT factor."""
    return float(np.exp(sum_a - sum_b))


def load_published_panel(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    if frame.isna().any().any():
        raise ValueError(f"{path} contains NaN")
    if "ACE2" not in frame.columns:
        raise ValueError(f"{path} has no ACE2 column")
    antibodies = frame.drop(columns=["ACE2"]).to_numpy(dtype=np.float64)
    return frame, antibodies


def sign_agreement(published: np.ndarray, predicted: np.ndarray, threshold: float = 0.3) -> float:
    mask = np.abs(published) > threshold
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(published[mask]) == np.sign(predicted[mask])))


def jaccard_disrupted(published: np.ndarray, predicted: np.ndarray, threshold: float = -0.3) -> float:
    left = np.nanmin(published, axis=0) < threshold
    right = np.nanmin(predicted, axis=0) < threshold
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def panel_comparison(published: np.ndarray, predicted: np.ndarray) -> dict:
    """Exploratory comparison. A weak match is not a bug: the published panel is TopNetmAb."""
    from topnet.metrics import regression_metrics

    flat_pub = published.ravel()
    flat_pred = predicted.ravel()
    return {
        **regression_metrics(flat_pub, flat_pred),
        "sign_agreement_abs_gt_0.3": sign_agreement(published, predicted, 0.3),
        "jaccard_disrupted_at_-0.3": jaccard_disrupted(published, predicted, -0.3),
        "disrupted_counts_predicted": disrupted_counts(predicted),
        "fold_reduction_predicted": fold_reduction(predicted),
        "note": "exploratory; antibodies_BFEs.csv is TopNetmAb, not an AB-Bind TopNetTree",
    }
