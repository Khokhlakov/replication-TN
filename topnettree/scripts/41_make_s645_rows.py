#!/usr/bin/env python3
"""Build the row table for the AB-Bind S645 H0 parity test (same columns that 42_build_scap_mutants.py and the generator use).

Reads data/labels/ab_bind_s645.csv (row order = row order of the shipped X_unnormed.npy / X_aux.npy), points every row at its
structure under data/structures/abbind/ and names the mutant file. Rows whose structure file is missing are reported and the
script exits with an error: nothing is guessed.

    python scripts/41_make_s645_rows.py [--labels data/labels/ab_bind_s645.csv] [--structures data/structures/abbind]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from topnet.sources import read_pdb_residues

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="data/labels/ab_bind_s645.csv")
ap.add_argument("--structures", default="data/structures/abbind")
ap.add_argument("--mutants", default="data/features/mutants")
ap.add_argument("--out", default="data/raw/s645_for_features.csv")
args = ap.parse_args()

lab = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
need = ["pdb", "chain", "wt", "resid", "mut"]
if any(c not in lab.columns for c in need):
    sys.exit(f"{args.labels} lacks one of {need}; columns are {list(lab.columns)}")
lab["row_idx"] = range(len(lab))
lab["structure"] = [str(Path(args.structures) / f"{p}.pdb") for p in lab["pdb"]]
miss = lab[~lab["structure"].map(lambda f: Path(f).is_file())]
if len(miss):
    print(miss[["row_idx", "pdb", "structure"]].drop_duplicates("pdb").head(20).to_string(index=False))
    sys.exit(f"{len(miss)} rows have no structure file under {args.structures}; fix the file names first")
lab["resid_struct"] = lab["resid"]
lab["mutant"] = [f"{args.mutants}/S645_{r.pdb}_{r.chain}_{r.wt}_{r.resid}_{r.mut}.pdb" for r in lab.itertuples()]

_cache = {}


def wt_ok(r):
    if r.structure not in _cache:
        _cache[r.structure] = {c: {k.upper(): v for k, v in d.items()} for c, d in read_pdb_residues(Path(r.structure)).items()}
    return int(_cache[r.structure].get(r.chain, {}).get(str(r.resid).upper()) == r.wt)


lab["wt_ok"] = [wt_ok(r) for r in lab.itertuples()]
print(f"wild-type letter matches the structure for {int(lab['wt_ok'].sum())} of {len(lab)} rows")
bad = lab[lab["wt_ok"] == 0]
if len(bad):
    print(bad[["row_idx", "pdb", "chain", "wt", "resid", "mut"]].head(20).to_string(index=False))

dup = lab["mutant"].duplicated(keep=False)
if dup.any():
    print(f"note: {int(lab['mutant'].duplicated().sum())} rows repeat an earlier (pdb, chain, wt, resid, mut); "
          "they share one mutant file and keep their own row_idx:")
    print(lab.loc[dup, ["row_idx", "pdb", "chain", "wt", "resid", "mut"]].to_string(index=False))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
lab.to_csv(args.out, index=False)
print(f"wrote {args.out}: {len(lab)} rows, {lab['pdb'].nunique()} structures")
