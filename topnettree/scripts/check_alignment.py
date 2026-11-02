#!/usr/bin/env python3
"""
check_alignment.py -- is the row order of X_unnormed.npy / X_aux.npy the same as the label table?

Three independent tests (none needs structures):

 A. Identical adjacent X_aux rows. Lists every adjacent pair of identical X_aux rows with its
    label annotation. Pairs that are the SAME mutation listed twice in AB-Bind (dup_group >= 0)
    are expected to be identical. Other pairs are unexplained; the script says whether X_unnormed
    (the H0 tensor) also has identical rows there.

 B. Complex-boundary test. If the arrays are in table order, many feature columns change at the
    complex boundaries (pdb changes) and far fewer inside a complex. Compares the fraction of
    changed columns at boundaries with the rest. Also checks the block structure
    (contiguous complexes) against the table.

 C. Within-complex correlation of CV predictions (optional --pred). If rows were aligned only at the
    COMPLEX level (e.g. a reordering inside each complex), the pooled r can be decent
    (complex means differ) while the within-complex r collapses toward zero.

Usage
  python scripts/check_alignment.py --labels data/labels/ab_bind_s645_supplement.csv \
       --aux data/raw/X_aux.npy --cnn data/raw/X_unnormed.npy
  python scripts/check_alignment.py ... --pred runs/<run>/predictions.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def pairs_report(L, A, X):
    idx = [i for i in range(len(A) - 1) if np.array_equal(A[i], A[i + 1])]
    print(f"[A] {len(idx)} identical adjacent X_aux pairs")
    rows = []
    for i in idx:
        a, b = L.iloc[i], L.iloc[i + 1]
        dup = bool(a.get("dup_group", -1) >= 0 and a.get("dup_group", -1) == b.get("dup_group", -2))
        same_site = (a.pdb == b.pdb and a.chain == b.chain and a.resid_str == b.resid_str)
        same_cx = a.pdb == b.pdb
        rows.append({
            "i": i, "pdb_i": a.pdb, "mut_i": f"{a.chain}:{a.wt}{a.resid_str}{a.mut}",
            "pdb_j": b.pdb, "mut_j": f"{b.chain}:{b.wt}{b.resid_str}{b.mut}",
            "duplicate_mutation": dup, "same_site": same_site, "same_complex": same_cx,
            "X_cnn_identical": bool(np.array_equal(X[i], X[i + 1])),
        })
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    print(f"    expected (duplicate mutation entries): {int(r.duplicate_mutation.sum())}")
    print(f"    same site, different mutant residue   : {int((r.same_site & ~r.duplicate_mutation).sum())}")
    print(f"    same complex, different site          : {int((r.same_complex & ~r.same_site).sum())}")
    print(f"    different complexes                   : {int((~r.same_complex).sum())}")
    print(f"    ... of which X_cnn also identical      : {int(r.X_cnn_identical.sum())}/{len(r)}")
    print(r.to_string(index=False))
    # run lengths of identical rows
    runs, k = [], 1
    for i in range(1, len(A)):
        if np.array_equal(A[i], A[i - 1]):
            k += 1
        else:
            if k > 1:
                runs.append(k)
            k = 1
    if k > 1:
        runs.append(k)
    print(f"    run lengths of identical aux rows: {sorted(set(runs))} (longest {max(runs) if runs else 0})")
    return r


def boundary_test(L, M, name):
    chg = (M[1:] != M[:-1]).mean(axis=1)            # fraction of columns changing between row i and i+1
    boundary = (L.pdb.values[1:] != L.pdb.values[:-1])
    nb, ni = int(boundary.sum()), int((~boundary).sum())
    fb, fi = chg[boundary].mean(), chg[~boundary].mean()
    # blocks contiguous?
    order = list(dict.fromkeys(L.pdb))
    contiguous = (len(order) == (L.pdb.values[1:] != L.pdb.values[:-1]).sum() + 1)
    # rank boundaries: how many of the top-nb change positions are true boundaries
    top = np.argsort(-chg)[:nb]
    hit = int(boundary[top].sum())
    print(f"[B:{name}] complexes contiguous in table: {contiguous}; boundaries={nb}")
    print(f"    mean fraction of columns changed: at boundaries {fb:.3f} vs inside complexes {fi:.3f}")
    print(f"    of the {nb} biggest changes, {hit} are true boundaries "
          f"(chance ~ {nb * nb / (nb + ni):.1f})")
    return hit, nb


def within_complex(L, pred_path):
    p = Path(pred_path)
    pr = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    need = {"row_idx", "y_true", "y_pred"}
    if not need <= set(pr.columns):
        print(f"[C] predictions file lacks {need - set(pr.columns)}; columns are {list(pr.columns)}")
        return
    pr = pr.merge(L[["row_idx", "complex_id"]], on="row_idx", how="left")
    keys = [c for c in ("seed",) if c in pr.columns]
    for k, g in (pr.groupby(keys) if keys else [("all", pr)]):
        g = g.copy()
        g["yt"] = g.y_true - g.groupby("complex_id").y_true.transform("mean")
        g["yp"] = g.y_pred - g.groupby("complex_id").y_pred.transform("mean")
        pooled = np.corrcoef(g.y_true, g.y_pred)[0, 1]
        within = np.corrcoef(g.yt, g.yp)[0, 1]
        print(f"[C] seed={k}: pooled r={pooled:.3f}   within-complex r={within:.3f}   n={len(g)}")
    print("    within-complex r near 0 while pooled r is decent => rows are aligned only at complex level.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels/ab_bind_s645_supplement.csv")
    ap.add_argument("--aux", default="data/raw/X_aux.npy")
    ap.add_argument("--cnn", default="data/raw/X_unnormed.npy")
    ap.add_argument("--pred", default=None, help="predictions CSV/parquet from a CV run (row_idx,y_true,y_pred[,seed])")
    a = ap.parse_args()
    L = pd.read_csv(a.labels, keep_default_na=False)
    A = np.load(a.aux)
    X = np.load(a.cnn).reshape(len(L), -1) if Path(a.cnn).exists() else None
    if len(A) != len(L):
        sys.exit(f"ERROR: {len(A)} aux rows vs {len(L)} labels")
    pairs_report(L, A, X if X is not None else A)
    boundary_test(L, A, "X_aux")
    if X is not None:
        boundary_test(L, X, "X_cnn")
    if a.pred:
        within_complex(L, a.pred)


if __name__ == "__main__":
    main()
