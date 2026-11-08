#!/usr/bin/env python3
"""Step 5. Score a (partial) 26 x 186 BFE table against numbers printed in the paper.

Rules, both verified on the shipped BA.1 rows:
  * "disrupted at threshold t" (Fig. 3a2/b2/c2/d2): the complex's WORST single-mutation BFE change is < t.
    This reproduces all 11 published BA.1 counts exactly. The paper text does not state the rule.
  * fold of affinity reduction (Fig. 3e): exp(-mean over antibodies of the sum of the negative parts),
    positives set to zero per mutation. Gives 20.94 for BA.1 (paper 20.9) and 1.29 for Alpha (paper 1.3).

With rows missing, adding mutations can only lower a complex's minimum and make the negative sum more
negative. So counts and folds computed from the known rows are LOWER BOUNDS on the paper's values. The report
says whether each bound is respected and how much the missing mutations still have to contribute.

    python scripts/50_check_paper_targets.py [--table data/mutation_sets/antibody_bfe_26x186.csv]
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
from topnet.mutations import VARIANTS  # noqa: E402

THRESHOLDS = [-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0]
# Fig. 3a2, b2, c2, d2 (read from the figure; the BA.1 list is reproduced exactly by the shipped data)
COUNTS = {
    "BA.1": [2, 5, 15, 33, 60, 78, 106, 124, 143, 163, 185],
    "BA.2": [2, 5, 16, 35, 65, 85, 115, 132, 150, 168, 185],
    "BA.3": [2, 5, 16, 34, 61, 79, 106, 123, 141, 164, 185],
    "Delta": [0, 0, 0, 5, 10, 16, 20, 32, 48, 70, 141],
}
# Fig. 3e (one decimal)
FOLDS = {"BA.1": 20.9, "BA.2": 27.1, "BA.3": 18.1, "Alpha": 1.3, "Beta": 4.2, "Gamma": 4.3, "Delta": 1.5, "Lambda": 2.5, "Mu": 2.3}
ACE2_SUMS = {"BA.1": 2.60, "BA.2": 2.98, "BA.3": 2.88}  # Fig. 1g / text
FOLD_TOL = 0.06


def evaluate(table: pd.DataFrame) -> list[dict]:
    antibodies = table.drop(columns=["ACE2"])
    results = []
    for variant, muts in VARIANTS.items():
        present = [m for m in muts if m in table.index and not table.loc[m].isna().any()]
        missing = [m for m in muts if m not in present]
        res: dict = {"variant": variant, "n_mut": len(muts), "missing": missing}
        if not present:
            res["status"] = "no data"
            results.append(res)
            continue
        block = antibodies.loc[present]
        worst = block.min(axis=0)
        counts = [int((worst < t).sum()) for t in THRESHOLDS]
        neg_mean = float(block.clip(upper=0).sum(axis=0).mean())
        fold = math.exp(-neg_mean)
        res.update(counts=counts, fold=round(fold, 3), complete=not missing)
        if variant in ACE2_SUMS:
            res["ace2_sum"] = round(float(table.loc[present, "ACE2"].sum()), 4)
        if variant in COUNTS:
            tgt = COUNTS[variant]
            if not missing:
                res["counts_match"] = counts == tgt
                res["counts_max_abs_err"] = int(max(abs(a - b) for a, b in zip(counts, tgt)))
            else:
                res["counts_bound_ok"] = all(c <= t for c, t in zip(counts, tgt))
                res["counts_still_needed"] = [t - c for c, t in zip(counts, tgt)]
        if variant in FOLDS:
            tgt = FOLDS[variant]
            if not missing:
                res["fold_ok"] = abs(fold - tgt) <= FOLD_TOL
            else:
                res["fold_bound_ok"] = fold <= tgt + FOLD_TOL
                res["mean_negative_still_needed"] = round(-math.log(tgt) - neg_mean, 3)
        if variant in ACE2_SUMS and not missing:
            res["ace2_ok"] = abs(res["ace2_sum"] - ACE2_SUMS[variant]) <= 0.006
        results.append(res)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/mutation_sets/antibody_bfe_26x186.csv")
    ap.add_argument("--json-out", default="data/mutation_sets/target_report.json")
    args = ap.parse_args()
    table = pd.read_csv(args.table, index_col="mutation")
    results = evaluate(table)
    print(f"thresholds: {THRESHOLDS}\n")
    for r in results:
        head = f"{r['variant']:6s} ({r['n_mut']} mutations)"
        if r.get("status") == "no data":
            print(f"{head}: no complete rows (missing {r['missing']})")
            continue
        state = "COMPLETE" if r["complete"] else f"PARTIAL, missing {r['missing']}"
        print(f"{head}: {state}\n    fold {r['fold']}" + (f" | ACE2 sum {r['ace2_sum']}" if "ace2_sum" in r else ""))
        if "counts_match" in r:
            print(f"    counts match paper: {r['counts_match']} (max abs error {r['counts_max_abs_err']})")
        if "counts_bound_ok" in r:
            print(f"    counts lower bound respected: {r['counts_bound_ok']}; antibodies still to be pushed below each threshold: "
                  f"{r['counts_still_needed']}")
        if "fold_ok" in r:
            print(f"    fold matches paper: {r['fold_ok']}")
        if "fold_bound_ok" in r:
            print(f"    fold lower bound respected: {r['fold_bound_ok']}; mean negative BFE the missing rows must add: "
                  f"{r['mean_negative_still_needed']}")
        if "ace2_ok" in r:
            print(f"    ACE2 sum matches paper: {r['ace2_ok']}")
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
