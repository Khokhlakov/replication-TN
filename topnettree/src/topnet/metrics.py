"""Pearson, Spearman, RMSE and MAE, including the non-binder exclusion."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _corr(fn, y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    result = fn(y, pred)
    if hasattr(result, "statistic"):
        return float(result.statistic)
    return float(result[0])


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return {
        "pearson_r": _corr(stats.pearsonr, y, pred),
        "spearman_rho": _corr(stats.spearmanr, y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))) if len(y) else float("nan"),
        "mae": float(mean_absolute_error(y, pred)) if len(y) else float("nan"),
        "n": int(len(y)),
    }


def metrics_with_nonbinders(y, pred, is_nonbinder) -> dict:
    mask = np.asarray(is_nonbinder).astype(bool)
    return {
        "all": regression_metrics(y, pred),
        "excluding_nonbinders": regression_metrics(np.asarray(y)[~mask], np.asarray(pred)[~mask]),
    }


def per_complex_metrics(y, pred, complex_id) -> list[dict]:
    y = np.asarray(y)
    pred = np.asarray(pred)
    complex_id = np.asarray(complex_id)
    rows = []
    for cid in list(dict.fromkeys(complex_id.tolist())):
        sel = complex_id == cid
        item = regression_metrics(y[sel], pred[sel])
        item["complex_id"] = cid
        rows.append(item)
    return rows


def mean_median_complex_r(rows: list[dict]) -> dict:
    values = np.array([r["pearson_r"] for r in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"mean_r": float("nan"), "median_r": float("nan"), "n_complexes": int(len(rows))}
    return {
        "mean_r": float(np.mean(finite)),
        "median_r": float(np.median(finite)),
        "n_complexes": int(len(rows)),
        "n_finite": int(len(finite)),
    }


def repeat_summary(records: list[dict]) -> dict:
    """Mean and sample standard deviation of a metric across CV repeats."""
    if not records:
        return {}
    keys = records[0].keys()
    out = {}
    for key in keys:
        values = np.array([r[key] for r in records], dtype=np.float64)
        out[key] = {
            "mean": float(np.nanmean(values)),
            "sd": float(np.nanstd(values, ddof=1)) if np.sum(np.isfinite(values)) > 1 else 0.0,
            "n": int(np.sum(np.isfinite(values))),
        }
    return out
