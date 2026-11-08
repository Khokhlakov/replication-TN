#!/usr/bin/env python3
"""Step 2. Compare regenerated features with the shipped ones, row by row, on the joined S1131 rows.

    python scripts/35_compare_features.py \\
        --shipped CAPSULE/code/crossvalidation/X_skempi_feature2.npy \\
        --generated data/features/s1131_aux/features_full.npy \\
        --rows data/raw/skempi/s1131_joined.csv  [--name aux]

--rows gives s1131_row for each generated row (generated row k  <->  shipped row rows.s1131_row[k]). NaN rows in the
generated matrix (failed generations) are skipped and counted.

Verdict ladder (first that holds): IDENTICAL (>=99.9% of entries equal within rtol 1e-5), SAME_UP_TO_PERMUTATION (>=95%
of informative shipped columns have a generated column with r>=0.999, at a different index), SAME_UP_TO_SCALE
(columns match with r>=0.999 at the same index but values differ), CORRELATED (median column r >= 0.9), DIFFERENT.
Writes data/features/compare_<name>.json and a per-column CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def colwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / den


def cross_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = (a - a.mean(0)) / np.where(a.std(0) > 0, a.std(0), np.nan)
    b = (b - b.mean(0)) / np.where(b.std(0) > 0, b.std(0), np.nan)
    return np.nan_to_num(a.T @ b / len(a), nan=0.0)


def compare(shipped: np.ndarray, gen: np.ndarray) -> tuple[dict, pd.DataFrame]:
    n, ds = shipped.shape
    dg = gen.shape[1]
    rep: dict = {"n_rows": int(n), "shipped_cols": int(ds), "generated_cols": int(dg)}
    informative = shipped.std(0) > 0
    rep["shipped_constant_cols"] = int((~informative).sum())
    if ds == dg:
        close = np.isclose(shipped, gen, rtol=1e-5, atol=1e-8)
        rep["entry_equal_frac"] = float(close.mean())
        rep["row_equal_frac"] = float(close.all(1).mean())
        r = colwise_corr(shipped, gen)
        r = np.where(informative, r, np.nan)
        rep["median_col_r"] = float(np.nanmedian(r))
        rep["frac_cols_r_ge_0.999"] = float(np.mean(r[informative] >= 0.999)) if informative.any() else 0.0
        sc, gc = shipped - shipped.mean(0), gen - gen.mean(0)
        slope = np.where(informative, (gc * sc).sum(0) / np.maximum((sc ** 2).sum(0), 1e-300), np.nan)
        percol = pd.DataFrame({"col": np.arange(ds), "r_same_index": r, "slope_gen_on_shipped": slope,
                               "entry_equal_frac": close.mean(0)})
    else:
        percol = pd.DataFrame({"col": np.arange(ds)})
        rep["entry_equal_frac"] = float("nan")
        rep["median_col_r"] = float("nan")
        rep["frac_cols_r_ge_0.999"] = 0.0
    cc = cross_corr(shipped[:, informative], gen)
    best_j = cc.argmax(1)
    best_r = cc.max(1)
    idx = np.flatnonzero(informative)
    percol.loc[idx, "best_gen_col"] = best_j
    percol.loc[idx, "best_gen_r"] = best_r
    rep["frac_shipped_cols_with_match_r_ge_0.999"] = float(np.mean(best_r >= 0.999)) if len(best_r) else 0.0
    rep["frac_matches_at_other_index"] = float(np.mean((best_r >= 0.999) & (best_j != idx))) if len(best_r) else 0.0
    if rep["entry_equal_frac"] >= 0.999:
        verdict = "IDENTICAL"
    elif rep["frac_shipped_cols_with_match_r_ge_0.999"] >= 0.95 and rep["frac_matches_at_other_index"] >= 0.5:
        verdict = "SAME_UP_TO_PERMUTATION"
    elif rep["frac_cols_r_ge_0.999"] >= 0.95:
        verdict = "SAME_UP_TO_SCALE"
    elif rep["median_col_r"] >= 0.9:
        verdict = "CORRELATED"
    else:
        verdict = "DIFFERENT"
    rep["verdict"] = verdict
    return rep, percol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--name", default="features")
    ap.add_argument("--out-dir", default="data/features")
    args = ap.parse_args()

    shipped = np.load(args.shipped, allow_pickle=True).astype(float)
    gen = np.load(args.generated, allow_pickle=True).astype(float)
    rows = pd.read_csv(args.rows)["s1131_row"].to_numpy()
    if len(rows) != len(gen):
        raise SystemExit(f"--rows has {len(rows)} entries but the generated matrix has {len(gen)} rows.")
    ok = ~np.isnan(gen).any(1)
    print(f"generated rows usable: {int(ok.sum())}/{len(gen)}")
    rep, percol = compare(shipped[rows[ok]], gen[ok])
    rep["n_failed_generation"] = int((~ok).sum())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compare_{args.name}.json").write_text(json.dumps(rep, indent=2))
    percol.to_csv(out / f"compare_{args.name}_columns.csv", index=False)
    print(json.dumps(rep, indent=2))
    print(f"VERDICT for {args.name}: {rep['verdict']}")


if __name__ == "__main__":
    main()
