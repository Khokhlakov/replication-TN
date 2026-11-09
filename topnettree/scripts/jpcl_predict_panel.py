#!/usr/bin/env python3
"""Predict BFE change = -ddg for the 26 x 186 panel and score the 15 BA.1 rows.

The feature array is in the same order as data/features/rows/panel.csv.
Non-finite rows stay NaN. antibodies_BFEs.csv holds the shipped model's
predictions. Agreement with that file is not a measured free energy.

Script 40 keeps the shipped BA.1 values when it fills the 11 missing names.
A later BA.1 count match from script 50 is the shipped table, not this model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from topnet.cnn import resolve_device  # noqa: E402
from topnet.config import load_config  # noqa: E402
from topnet.mutations import BA1_ROW_ORDER, FIG2_ORDER  # noqa: E402
from topnet.pipeline import load_bundle, predict_pipeline  # noqa: E402


def correlation(left: np.ndarray, right: np.ndarray) -> dict:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return {"n": int(mask.sum()), "pearson": None, "spearman": None}
    return {
        "n": int(mask.sum()),
        "pearson": float(stats.pearsonr(left[mask], right[mask])[0]),
        "spearman": float(stats.spearmanr(left[mask], right[mask])[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--rows", default="data/features/rows/panel.csv")
    parser.add_argument("--h0", required=True)
    parser.add_argument("--bfes", default="data/raw/antibodies_BFEs.csv")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    panel = pd.read_csv(args.rows)
    array = np.load(args.h0)
    if array.ndim == 2 and array.shape[1] == 5184:
        array = array.reshape(len(array), 48, 108)
    if len(array) != len(panel):
        raise SystemExit(f"{args.h0} has {len(array)} rows and {args.rows} has {len(panel)}")
    ok = np.isfinite(array).all(axis=(1, 2))
    bundle = load_bundle(args.bundle)
    if bundle.get("label") != "no-H12":
        raise SystemExit(f"bundle label is {bundle.get('label')}")
    cfg = load_config(args.config)
    pred = np.full(len(panel), np.nan)
    pred[ok] = predict_pipeline(bundle, array[ok], device=resolve_device(cfg.get("device", "auto")))
    panel = panel.assign(ddg=pred, bfe_change=-pred)
    ship = pd.read_csv(args.bfes)
    wide = panel.pivot(index="mutation", columns="label", values="bfe_change")
    missing_cols = [column for column in ship.columns if column not in wide.columns]
    if missing_cols:
        raise SystemExit(f"panel labels missing from the BFE header: {missing_cols[:5]}")
    wide = wide.reindex(index=FIG2_ORDER, columns=list(ship.columns))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out)
    panel.to_csv(out.with_name(out.stem + "_long.csv"), index=False)
    known = wide.loc[BA1_ROW_ORDER]
    per_mutation = [{"mutation": mutation, **correlation(known.loc[mutation].to_numpy(), ship.iloc[i].to_numpy())}
                    for i, mutation in enumerate(BA1_ROW_ORDER)]
    per_antibody = [{"antibody": column, **correlation(known[column].to_numpy(), ship[column].to_numpy())}
                    for column in ship.columns]
    summary = {
        "label": bundle["label"],
        "n_cells": int(wide.size),
        "n_finite": int(np.isfinite(wide.to_numpy()).sum()),
        "ba1_overall": correlation(known.to_numpy().ravel(), ship.to_numpy().ravel()),
        "per_mutation": per_mutation,
        "per_antibody": per_antibody,
        "note": "Compared with shipped model predictions in antibodies_BFEs.csv. BFE change = -ddg.",
    }
    out.with_suffix(".ba1_agreement.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["ba1_overall"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
