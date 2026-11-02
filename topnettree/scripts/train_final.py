"""Fit deployment models on all S645 rows.

This is not a cross-validation score. Without X_h12 the saved models are the
cnn_aux, cnn and aux ablations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from topnet.config import load_config
from topnet.data import assert_label_tables, load_arrays
from topnet.pipeline import fit_pipeline, save_bundle
from topnet.runs import start_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--h12", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    x_cnn, x_aux, _idx = load_arrays(cfg)
    if args.synthetic:
        y = np.random.default_rng(int(cfg["seed"])).normal(size=len(x_cnn))
        meta_rows = np.arange(len(y))
    else:
        y, meta, _sup = assert_label_tables(cfg)
        meta_rows = meta["row_idx"].to_numpy()
    x_h12 = np.load(args.h12) if args.h12 else None
    feature_sets = ["cnn_aux", "cnn", "aux"]
    if x_h12 is not None:
        feature_sets = ["cnn_aux_h12", "cnn_h12", "h12"] + feature_sets
    run_dir = start_run("final", cfg, synthetic=bool(args.synthetic))
    for feature_set in feature_sets:
        bundle = fit_pipeline(
            x_cnn,
            y,
            None if feature_set == "cnn" else x_aux,
            cfg,
            seed=int(cfg["seed"]),
            feature_set=feature_set,
            row_idx=meta_rows,
            x_h12=x_h12,
        )
        name = f"SYNTHETIC_bundle_{feature_set}.joblib" if args.synthetic else f"bundle_{feature_set}.joblib"
        save_bundle(bundle, run_dir / name)
        print(feature_set, bundle["label"], "aux>|1e6|", bundle["aux_values_above_1e6"])
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
