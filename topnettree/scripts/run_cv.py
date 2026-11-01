"""Run nested or leaky cross-validation and write runs/<timestamp>_<name>/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from topnet.config import load_config
from topnet.cv import run_cv
from topnet.data import load_arrays, load_labels
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
    parser.add_argument("--synthetic", action="store_true", help="Random labels. Outputs are stamped SYNTHETIC.")
    parser.add_argument("--h12", default=None, help="Optional X_h12.npy with one row per S645 sample.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.mode:
        cfg["cv"]["mode"] = args.mode
    if args.feature_set:
        cfg["cv"]["feature_set"] = args.feature_set
    x_cnn, x_aux, idx_ref = load_arrays(cfg)
    synthetic = bool(args.synthetic)
    if synthetic:
        y = _synthetic_labels(len(x_cnn), int(cfg["seed"]))
        meta = None
        row_idx = np.arange(len(y))
    else:
        y, meta = load_labels(cfg=cfg)
        row_idx = meta["row_idx"].to_numpy()
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
