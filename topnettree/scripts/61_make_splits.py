#!/usr/bin/env python3
"""Step 8. Split definitions for the training stage, written once and frozen with the data.

Adds columns to data/labels/reconstructed_labels.csv (written to data/splits/labels_with_folds.csv):

  in_s1131               SKEMPI row is one of the 1131 rows with shipped features (needs 34_join_s1131.py output)
  ab_bind_pdb_overlap    SKEMPI row lies on a PDB that also holds AB-Bind S645 rows (from 10_fetch_skempi.py)
  excl_for_s645_eval     1 = must NOT be used to train a model that is evaluated on AB-Bind S645 (all SKEMPI rows on S645 PDBs)
  group                  leakage group: SKEMPI = PDB id, with PDBs merged when they hold reverse mutations of each other
                         (A->B in one entry, B->A in another, ddG exactly negated; e.g. 3MZG with 3NCB/3NCC); DMS = residue position
  fold_row_s<seed>       SKEMPI only: ordinary shuffled K-fold over rows (comparable to the paper's 10-fold CV)
  fold_group_s<seed>     all datasets: K-fold over groups (no group straddles train and test). This is the honest number.
  protocol_A             SI S5 protocol: SKEMPI = train, Starr/Chan = test (held-out DMS)

Rules the training stage must follow:
  * Report both fold_row and fold_group results; they answer different questions.
  * Anything evaluated on AB-Bind S645 is trained with excl_for_s645_eval == 0 only.
  * Protocol A never trains on DMS rows. Protocol B (final model) may, and its DMS numbers are then not validation.

    python scripts/61_make_splits.py [--seeds 0 1 2] [--folds 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def find(parent: dict, x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def reverse_pairs(sk: pd.DataFrame, tol: float = 1e-3) -> list[tuple[str, str]]:
    """PDB pairs holding exact reverse mutations: same chain and residue, wt/mut swapped, ddG negated."""
    idx = {}
    for r in sk.itertuples():
        idx.setdefault((r.chain, r.resid, r.wt, r.mut), []).append((r.pdb, r.ddg))
    pairs = set()
    for (chain, resid, wt, mut), items in idx.items():
        for pdb_b, ddg_b in idx.get((chain, resid, mut, wt), []):
            for pdb_a, ddg_a in items:
                if pdb_a != pdb_b and abs(ddg_a + ddg_b) < tol:
                    pairs.add(tuple(sorted((pdb_a, pdb_b))))
    return sorted(pairs)


def balanced_group_folds(groups: pd.Series, k: int, seed: int) -> pd.Series:
    """Shuffle groups, then give each to the currently smallest fold. Deterministic for a seed."""
    sizes = groups.value_counts()
    order = list(sizes.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    order.sort(key=lambda g: -sizes[g])  # big groups first, ties keep the shuffled order (stable sort)
    load = np.zeros(k, dtype=int)
    assign = {}
    for g in order:
        f = int(load.argmin())
        assign[g] = f
        load[f] += sizes[g]
    return groups.map(assign)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/reconstructed_labels.csv")
    ap.add_argument("--skempi", default="data/raw/skempi/skempi_v2_single_labels.csv")
    ap.add_argument("--s1131", default="data/raw/skempi/s1131_joined.csv")
    ap.add_argument("--out", default="data/splits/labels_with_folds.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    lab["ddg"] = pd.to_numeric(lab["ddg"])
    sk = pd.read_csv(args.skempi, dtype=str, keep_default_na=False)
    sk["ddg"] = pd.to_numeric(sk["ddg"])
    sk["resid_merge"] = sk["resid_struct"] if "resid_struct" in sk.columns else sk["resid"]  # same choice as 60_merge_labels.py
    sk["k_labels"] = sk["pdb"] + "|" + sk["chain"] + "|" + sk["wt"] + "|" + sk["resid_merge"] + "|" + sk["mut"]

    # S1131 membership
    sk["in_s1131"] = 0
    s1131 = Path(args.s1131)
    if s1131.is_file():
        j = pd.read_csv(s1131, dtype=str, keep_default_na=False)
        rep_path = s1131.with_name("s1131_join_report.json")
        scheme = json.loads(rep_path.read_text()).get("best_scheme", "cleaned") if rep_path.is_file() else "cleaned"
        col = "resid" if scheme == "cleaned" else "resid_pdb"
        keys = set(j["pdb"] + "|" + j["chain_s"] + "|" + j["wt"] + "|" + j["resid_s"].str.upper() + "|" + j["mut"])
        sk["in_s1131"] = (sk["pdb"].str.upper() + "|" + sk["chain"] + "|" + sk["wt"] + "|" + sk[col].str.upper() + "|" + sk["mut"]).isin(keys).astype(int)
        print(f"S1131 rows found in the SKEMPI singles: {int(sk['in_s1131'].sum())} (scheme {scheme})")
    else:
        print(f"(no {s1131}: in_s1131 left at 0)")
    sk["ab_bind_pdb_overlap"] = pd.to_numeric(sk["ab_bind_pdb_overlap"]) if "ab_bind_pdb_overlap" in sk.columns else 0

    # leakage groups for SKEMPI: PDB ids merged across reverse-mutation structures
    parent: dict = {}
    pairs = reverse_pairs(sk)
    for a, b in pairs:
        parent[find(parent, a)] = find(parent, b)
    sk["group"] = [find(parent, p) for p in sk["pdb"]]
    print(f"reverse-mutation PDB pairs merged into shared groups: {pairs}")

    is_sk = lab["dataset"] == "skempi2"
    key_lab = lab["pdb"] + "|" + lab["chain"] + "|" + lab["wt"] + "|" + lab["resid"] + "|" + lab["mut"]
    info = sk.drop_duplicates("k_labels").set_index("k_labels")[["in_s1131", "ab_bind_pdb_overlap", "group"]]
    lab["in_s1131"] = 0
    lab["ab_bind_pdb_overlap"] = 0
    lab["group"] = lab["dataset"] + "|" + lab["resid"]  # DMS default: residue position
    m = key_lab[is_sk].map(lambda k: k in info.index)
    if not m.all():
        raise SystemExit(f"{int((~m).sum())} SKEMPI label rows have no match in {args.skempi}; was 60_merge_labels.py run on this table?")
    lab.loc[is_sk, "in_s1131"] = info.loc[key_lab[is_sk], "in_s1131"].to_numpy()
    lab.loc[is_sk, "ab_bind_pdb_overlap"] = info.loc[key_lab[is_sk], "ab_bind_pdb_overlap"].to_numpy()
    lab.loc[is_sk, "group"] = "sk|" + info.loc[key_lab[is_sk], "group"].to_numpy()
    lab["excl_for_s645_eval"] = ((lab["dataset"] == "skempi2") & (lab["ab_bind_pdb_overlap"].astype(int) == 1)).astype(int)
    lab["protocol_A"] = np.where(is_sk, "train", "test")

    for seed in args.seeds:
        lab[f"fold_row_s{seed}"] = -1
        idx = np.flatnonzero(is_sk)
        for f, (_, te) in enumerate(KFold(args.folds, shuffle=True, random_state=seed).split(idx)):
            lab.iloc[idx[te], lab.columns.get_loc(f"fold_row_s{seed}")] = f
        lab[f"fold_group_s{seed}"] = -1
        for ds, part in lab.groupby("dataset"):
            lab.loc[part.index, f"fold_group_s{seed}"] = balanced_group_folds(part["group"], args.folds, seed).to_numpy()

    # checks
    for seed in args.seeds:
        c = f"fold_group_s{seed}"
        straddle = lab.groupby("group")[c].nunique()
        assert (straddle == 1).all(), f"group straddles folds for seed {seed}"
    print("check passed: no leakage group straddles a fold for any seed")
    s = args.seeds[0]
    print(f"\nfold sizes (seed {s}):")
    print(pd.crosstab(lab["dataset"], lab[f"fold_group_s{s}"]).to_string())
    print(f"\nSKEMPI rows excluded when evaluating on AB-Bind S645: {int(lab.excl_for_s645_eval.sum())} of {int(is_sk.sum())}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(lab)} rows, {lab.shape[1]} columns)")


if __name__ == "__main__":
    main()
