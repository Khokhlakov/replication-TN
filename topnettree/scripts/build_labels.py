"""Write data/labels/ab_bind_s645.csv from the AB-Bind S645 table.

Row order is the file order. It matches code/crossvalidation/Y.npy exactly when
that array is present, which is the evidence that this order is the training order.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from topnet.config import repo_root
from topnet.data import aux_alignment_report, labels_from_ab_bind, load_raw_aux

AB_CSV = repo_root().parent / "capsule-2202829" / "data" / "AB-Bind-Database" / "AB_645.csv"
Y_NPY = repo_root().parent / "capsule-2202829" / "code" / "crossvalidation" / "Y.npy"
OUT = repo_root() / "data" / "labels" / "ab_bind_s645.csv"


def main() -> None:
    if not AB_CSV.is_file():
        raise SystemExit(f"AB-Bind table not found: {AB_CSV}")
    frame = labels_from_ab_bind(AB_CSV)
    if Y_NPY.is_file():
        raw = np.load(Y_NPY, allow_pickle=True)
        y = np.array([float(v.decode() if isinstance(v, (bytes, bytearray)) else v) for v in raw])
        if not np.allclose(y, frame["ddg"].to_numpy()):
            raise SystemExit("AB_645 ddG does not match crossvalidation/Y.npy. Refusing to write labels.")
        print("Y.npy matches AB_645 ddG in file order.")
    else:
        print(f"Y.npy not found at {Y_NPY}; wrote the table order without that check.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    report = aux_alignment_report(load_raw_aux(), frame)
    print(f"wrote {OUT}")
    print(
        "identical aux pairs: {n_identical_adjacent_pairs}, "
        "same complex+chain+resid: {same_complex_chain_resid}, "
        "same complex: {same_complex}, zero-vector pairs: {zero_vector_pairs}, "
        "site match rate: {site_match_rate:.3f}".format(**report)
    )


if __name__ == "__main__":
    main()
