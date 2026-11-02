"""How many of the 3 duplicate groups straddle an outer kfold split.

Uses the same KFold generator as src/topnet/cv.py. Does not write new fold ids.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from topnet.config import load_config
from topnet.cv import kfold_splits
from topnet.data import assert_label_tables


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/legacy_ab_cv.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _y, _meta, sup = assert_label_tables(cfg)
    groups = sup.loc[sup["dup_group"].astype(int) >= 0, ["row_idx", "dup_group", "pdb"]]
    n_groups = groups["dup_group"].nunique()
    if n_groups != 3:
        raise AssertionError(f"expected 3 dup groups, found {n_groups}")
    seeds = [int(seed) for seed in cfg["cv"]["seeds"]]
    n_splits = int(cfg["cv"]["n_splits"])
    print(f"dup groups: {n_groups}. seeds {seeds}. n_splits {n_splits}. generator kfold_splits.")
    total = 0
    for seed in seeds:
        splits = kfold_splits(len(sup), n_splits, seed)
        fold_of = np.full(len(sup), -1, dtype=np.int64)
        for fold, (_train, test) in enumerate(splits):
            fold_of[test] = fold
        straddling = 0
        for group, part in groups.groupby("dup_group"):
            folds = sorted(set(fold_of[part["row_idx"].to_numpy()].tolist()))
            split = len(folds) > 1
            straddling += int(split)
            print(f"seed {seed} dup_group {int(group)} pdb {part['pdb'].iloc[0]} folds {folds} straddle {split}")
        print(f"seed {seed}: {straddling} of {n_groups} duplicate groups straddle outer folds")
        total += straddling
    print(f"sum over seeds of straddling groups: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
