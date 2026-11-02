"""Run nested or leaky cross-validation and write runs/<timestamp>_<name>/."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from topnet.config import load_config
from topnet.cv import run_cv
from topnet.data import assert_label_tables, load_arrays
from topnet.runs import start_run, write_outputs
from topnet.select import save_bin_profile_plot


def _synthetic_labels(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-validate the no-H12 ablations or a with-H12 model.")
    parser.add_argument("--config", default="configs/fast.yaml")
    parser.add_argument("--mode", choices=["nested", "leaky_original"], default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Overrides cv.seeds without changing the fold generator.")
    parser.add_argument("--synthetic", action="store_true", help="Random labels. Outputs are stamped SYNTHETIC.")
    parser.add_argument("--h12", default=None, help="Optional X_h12.npy with one row per S645 sample.")
    parser.add_argument(
        "--row-subset",
        default=None,
        help="Text file of row_idx values to keep. Fold ids stay on the 645-row assignment; other rows leave both train and test.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.get("tune_status") == "not_run" and cfg["cv"]["splitter"] != "kfold10":
        print(
            "WARNING: configs/tuned.yaml has tune_status not_run. "
            "The GBDT is the fast_nested default, not a T2 search result."
        )
    if args.mode:
        cfg["cv"]["mode"] = args.mode
    if args.feature_set:
        cfg["cv"]["feature_set"] = args.feature_set
    if args.seeds:
        cfg["cv"]["seeds"] = [int(item) for item in args.seeds.split(",")]
    x_cnn, x_aux, idx_ref = load_arrays(cfg)
    synthetic = bool(args.synthetic)
    if synthetic:
        y = _synthetic_labels(len(x_cnn), int(cfg["seed"]))
        meta = None
        row_idx = np.arange(len(y))
    else:
        y, meta, sup = assert_label_tables(cfg)
        meta = meta.copy()
        meta["parent_pdb"] = sup["parent_pdb"].astype(str).to_numpy()
        meta["dup_group"] = sup["dup_group"].to_numpy()
        meta["is_homology_model"] = sup["is_homology_model"].to_numpy()
        row_idx = meta["row_idx"].to_numpy()
        if int(meta["is_homology"].sum()) != 87:
            raise AssertionError(f"expected 87 homology rows, found {int(meta['is_homology'].sum())}")
        if meta["complex_id"].nunique() != 29:
            raise AssertionError(f"expected 29 complexes, found {meta['complex_id'].nunique()}")
    row_keep = None
    if args.row_subset:
        keep_ids = np.loadtxt(args.row_subset, dtype=np.int64).reshape(-1)
        row_keep = np.isin(row_idx, keep_ids)
    x_h12 = np.load(args.h12) if args.h12 else None
    name = f"{cfg['cv']['mode']}_{cfg['cv']['feature_set']}_{cfg['cv']['splitter']}"
    run_dir = start_run(name, cfg, synthetic=synthetic)
    result = run_cv(
        x_cnn,
        y,
        x_aux,
        cfg,
        row_idx=row_idx,
        meta=meta,
        x_h12=x_h12,
        cv_mode=cfg["cv"]["mode"],
        feature_set=cfg["cv"]["feature_set"],
        row_keep=row_keep,
    )
    order = result.pop("importance_order")
    result.pop("fit_log")
    if order is not None:
        save_bin_profile_plot(idx_ref, order, run_dir / "bin_profile.png")
    predictions = result.pop("predictions")
    if synthetic:
        predictions.to_csv(run_dir / "SYNTHETIC_predictions.csv", index=False)
    write_outputs(run_dir, predictions, result)
    print(result["label"])
    print(result["summary"])
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
