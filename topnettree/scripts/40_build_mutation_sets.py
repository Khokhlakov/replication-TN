#!/usr/bin/env python3
"""Step 4. Mutation sets, the 26 x 186 BFE table with its holes marked, and what the paper text pins down.

What is recoverable without a trained TopNetMAb:
  * the full mutation/variant definitions (from the paper text and Fig. 2 caption),
  * the 15 BA.1 rows already in antibodies_BFEs.csv, placed in their Fig. 2 positions,
  * exact interval constraints on the missing ACE2 values, derived from the three accumulated BFE sums
    printed in the paper (BA.1 2.60, BA.2 2.98, BA.3 2.88) plus the Delta infectivity ratio (4.2).

What is NOT recoverable here: the BFE changes of the 11 missing mutations on the 185 antibodies. They need the
model applied to structures. This script leaves them as NaN. Fill them with --predictions once a model exists,
then run 50_check_paper_targets.py to see whether the filled table reproduces the paper's own numbers.

The constraint derivation relies on the paper's definition of the accumulated BFE change as a plain sum over
RBD mutations, and on infectivity ratio = exp(delta of summed ACE2 BFE) (this reproduces 20, 4.2 and 1.5 from
2.98, 1.55 and 2.60; it is inferred, not stated in the text).

    python scripts/40_build_mutation_sets.py [--bfes data/raw/antibodies_BFEs.csv] [--predictions FILE]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.mutations import (  # noqa: E402
    BA1_ONLY, BA1_ROW_ORDER, BA2_ONLY, BA3_ONLY, EARLIER7, FIG2_ORDER, MISSING_11, SHARED12, VARIANTS, split,
)

ACE2_SUM_BA1 = 2.6018  # what the shipped file sums to; the paper prints 2.60
HALF = 0.005  # two-decimal rounding of the figures printed in the paper


def mutation_table() -> pd.DataFrame:
    rows = []
    for i, m in enumerate(FIG2_ORDER):
        wt, pos, mut = split(m)
        group = "earlier_variants" if m in EARLIER7 else "shared_omicron" if m in SHARED12 else "ba1_specific" if m in BA1_ONLY else "ba2_specific"
        row = {"fig2_order": i, "mutation": m, "wt": wt, "resid": pos, "mut": mut, "group": group,
               "in_shipped_bfe_file": int(m in BA1_ROW_ORDER)}
        row.update({f"in_{k}": int(m in v) for k, v in VARIANTS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def load_known(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.shape[0] != len(BA1_ROW_ORDER):
        sys.exit(f"{path} has {raw.shape[0]} rows, expected {len(BA1_ROW_ORDER)}")
    ace2 = float(raw["ACE2"].sum())
    if abs(ace2 - ACE2_SUM_BA1) > 5e-4:
        sys.exit(f"ACE2 column sums to {ace2:.4f}, expected {ACE2_SUM_BA1}: the row order assumption fails")
    raw.index = BA1_ROW_ORDER
    return raw


def ace2_constraints(known: pd.DataFrame) -> dict:
    a = known["ACE2"]
    s12 = float(a[SHARED12].sum())
    g446s = float(a["G446S"])
    t478k = float(a["T478K"])
    ba2 = (2.98 - HALF, 2.98 + HALF)
    ba3 = (2.88 - HALF, 2.88 + HALF)
    # S371F + T376A + D405N + R408S
    ba2_only = (ba2[0] - s12, ba2[1] - s12)
    # S371F + D405N (BA.3 = shared + S371F + D405N + G446S)
    ba3_pair = (ba3[0] - s12 - g446s, ba3[1] - s12 - g446s)
    # T376A + R408S
    rest = (ba2_only[0] - ba3_pair[1], ba2_only[1] - ba3_pair[0])
    # Delta: exp(BA.2 - Delta) = 4.2 (two significant figures: 4.15..4.25)
    delta = (ba2[0] - math.log(4.25), ba2[1] - math.log(4.15))
    l452r = (delta[0] - t478k, delta[1] - t478k)
    checks = {
        "infectivity_BA2_vs_original": [math.exp(ba2[0]), math.exp(ba2[1]), "paper: about 20"],
        "infectivity_BA2_vs_BA1": [math.exp(ba2[0] - ACE2_SUM_BA1), math.exp(ba2[1] - ACE2_SUM_BA1), "paper: about 1.5"],
    }
    return {
        "known": {"shared12_sum": s12, "G446S": g446s, "T478K": t478k, "BA1_sum": float(a.sum())},
        "intervals_kcal_per_mol": {
            "S371F+T376A+D405N+R408S": ba2_only,
            "S371F+D405N": ba3_pair,
            "T376A+R408S": rest,
            "L452R": l452r,
        },
        "consistency_checks": checks,
        "note": "Intervals assume additivity of per-mutation ACE2 BFE (the paper's definition of accumulated BFE) and "
                "are widened only by two-decimal rounding of the printed sums and the ratio 4.2.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfes", default="data/raw/antibodies_BFEs.csv")
    ap.add_argument("--out-dir", default="data/mutation_sets")
    ap.add_argument("--predictions", help="CSV with a 'mutation' column and the same 186 antibody columns")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    known = load_known(Path(args.bfes))
    table = mutation_table()
    table.to_csv(out / "mutations_26.csv", index=False)
    pd.DataFrame({"variant": list(VARIANTS), "n_rbd_mutations": [len(v) for v in VARIANTS.values()],
                  "mutations": [" ".join(v) for v in VARIANTS.values()]}).to_csv(out / "variant_sets.csv", index=False)

    full = known.reindex(FIG2_ORDER)
    if args.predictions:
        pred = pd.read_csv(args.predictions).set_index("mutation")
        missing_cols = set(full.columns) - set(pred.columns)
        if missing_cols:
            sys.exit(f"predictions lack columns: {sorted(missing_cols)[:5]}")
        overlap = [m for m in pred.index if m in BA1_ROW_ORDER]
        if overlap:
            diff = float((pred.loc[overlap, full.columns] - known.loc[overlap]).abs().to_numpy().max())
            print(f"predictions also contain {len(overlap)} shipped rows; max abs difference {diff:.3f} (shipped values kept)")
        for m in MISSING_11:
            if m in pred.index:
                full.loc[m] = pred.loc[m, full.columns]
    full.index.name = "mutation"
    full.to_csv(out / "antibody_bfe_26x186.csv")
    n_missing = int(full.isna().any(axis=1).sum())
    print(f"mutation table: {len(table)} rows | BFE table {full.shape} | rows still missing: {n_missing} "
          f"{[m for m in FIG2_ORDER if full.loc[m].isna().any()]}")

    cons = ace2_constraints(known)
    (out / "ace2_constraints.json").write_text(json.dumps(cons, indent=2))
    print("\nACE2 constraints on the missing mutations (kcal/mol, additive sums):")
    for k, (lo, hi) in cons["intervals_kcal_per_mol"].items():
        print(f"  {k:26s} [{lo:+.3f}, {hi:+.3f}]")
    for k, (lo, hi, note) in cons["consistency_checks"].items():
        print(f"  {k:30s} {lo:.2f}..{hi:.2f}   ({note})")
    print(f"\nwrote {out}/mutations_26.csv, variant_sets.csv, antibody_bfe_26x186.csv, ace2_constraints.json")


if __name__ == "__main__":
    main()
