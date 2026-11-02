"""Describe the (48, 108) H0 tensor. Run from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from topnet.data import load_importance_index, load_raw_aux, load_raw_cnn
from topnet.select import bin_profile

BLOCK_NAMES = (
    "binding death counts",
    "binding persistence counts",
    "mutation-site death counts",
    "mutation-site persistence counts",
)
PAIR_NAMES = ("CC", "CN", "CO", "NC", "NN", "NO", "OC", "ON", "OO")


def channel_means(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=(0, 1))


def describe(x: np.ndarray) -> str:
    means = channel_means(x)
    lines = [
        f"shape {x.shape} dtype {x.dtype} min {x.min()} max {x.max()}",
        "Layout verified from gen_cnn_feature.py, not only from the column means:",
        "48 bins of 0.25 A on [0, 12). Four blocks of 27 channels.",
        "Inside each block: 9 element pairs x (WT, MUT, WT-MUT).",
        "Element order: C, N, O against C, N, O.",
        "",
    ]
    for block, name in enumerate(BLOCK_NAMES):
        start = block * 27
        wt = means[start : start + 9]
        mut = means[start + 9 : start + 18]
        diff = means[start + 18 : start + 27]
        lines.append(f"block {block} ({name}) columns {start}:{start + 27}")
        lines.append("  WT  " + " ".join(f"{v:8.3f}" for v in wt))
        lines.append("  MUT " + " ".join(f"{v:8.3f}" for v in mut))
        lines.append("  DIFF " + " ".join(f"{v:8.3f}" for v in diff))
        lines.append(
            f"  max |WT-MUT| {np.max(np.abs(wt - mut)):.4f}  max |DIFF| {np.max(np.abs(diff)):.4f}"
        )
    std = x.std(axis=0)
    zero = std == 0
    lines.append("")
    lines.append(
        f"zero-std cells: {int(zero.sum())} / {zero.size} = {100 * zero.mean():.2f}%"
        f"; all identically 0: {bool(np.all(x[:, zero] == 0))}"
    )
    return "\n".join(lines)


def main() -> None:
    x = load_raw_cnn()
    aux = load_raw_aux()
    idx = load_importance_index()
    print(describe(x))
    print("element pairs:", " ".join(PAIR_NAMES))
    profile = bin_profile(idx, 300)
    print("reference top-300 count by bin:")
    print(" ".join(f"{b}:{int(c)}" for b, c in enumerate(profile) if c))
    same = np.all(aux[1:] == aux[:-1], axis=1)
    print(f"identical adjacent aux pairs: {int(same.sum())}")
    const = np.where(aux.std(axis=0) == 0)[0]
    print(f"constant aux columns: {const.tolist()} count {len(const)}")
    from topnet.data import assert_label_tables
    from topnet.config import load_config

    _y, meta, _sup = assert_label_tables(load_config("configs/default.yaml"))
    print("distinct X_aux rows per complex:")
    for pdb, part in meta.groupby("pdb", sort=False):
        rows = part["row_idx"].to_numpy()
        block = aux[rows]
        distinct = np.unique(block, axis=0).shape[0]
        print(f"  {pdb} rows {len(rows)} distinct {distinct}")
    big = np.where(np.max(np.abs(aux), axis=0) > 1e6)[0]
    print(f"aux columns with |x|>1e6: {len(big)}; max {np.max(np.abs(aux)):.6e}")


if __name__ == "__main__":
    main()
