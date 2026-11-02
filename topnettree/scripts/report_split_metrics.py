"""Read a prediction CSV and report homology or LOCO numbers.

Non-binders use is_nonbinder. Bootstrap intervals are over test rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from topnet.config import load_config
from topnet.data import assert_label_tables
from topnet.metrics import metrics_with_nonbinders, per_complex_metrics


def _bootstrap_r(y, pred, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    scores = []
    n = len(y)
    for _ in range(n_boot):
        take = rng.integers(0, n, n)
        if np.std(y[take]) == 0 or np.std(pred[take]) == 0:
            continue
        scores.append(float(np.corrcoef(y[take], pred[take])[0, 1]))
    if not scores:
        return float("nan"), float("nan")
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--group", choices=["complex_id", "parent_pdb"], default="complex_id")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    cfg = load_config("configs/default.yaml")
    _y, meta, sup = assert_label_tables(cfg)
    pred = pd.read_csv(args.predictions)
    pred = pred[np.isfinite(pred["y_pred"])].copy()
    meta = meta.copy()
    meta["parent_pdb"] = sup["parent_pdb"].astype(str).to_numpy()
    merged = pred.merge(meta, on="row_idx", suffixes=("", "_label"))
    flag = merged["is_nonbinder"].to_numpy()
    both = metrics_with_nonbinders(merged["y_true"], merged["y_pred"], flag)
    low, high = _bootstrap_r(merged["y_true"], merged["y_pred"], args.bootstrap, seed=0)
    print("pooled", both["all"])
    print(f"bootstrap Pearson r 95% interval over rows: {low:.4f} {high:.4f} (n_boot={args.bootstrap})")
    print("excluding non-binders", both["excluding_nonbinders"])
    group = merged[args.group].astype(str).to_numpy()
    rows = per_complex_metrics(merged["y_true"], merged["y_pred"], group)
    frame = pd.DataFrame(rows)
    finite = frame[np.isfinite(frame["pearson_r"])]
    print(f"per-{args.group} finite r: n={len(finite)} mean={finite['pearson_r'].mean():.4f} median={finite['pearson_r'].median():.4f}")
    keep = frame[(frame["n"] >= 5) & np.isfinite(frame["pearson_r"])]
    print(f"excluding n<5 or undefined r: n={len(keep)} mean={keep['pearson_r'].mean() if len(keep) else float('nan'):.4f}")
    print("The 0.55 and 0.51 targets are for the full published model. A miss without H12 is not a failure of this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
