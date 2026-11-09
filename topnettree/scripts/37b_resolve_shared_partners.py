#!/usr/bin/env python3
"""Step 3c. Resolve the 37_freeze_complex_map.py QC rows that failed only with 'shared_partner_unresolved'.

An entry with several RBD copies often has antibody chains that touch more than one copy (crystal packing, or a Fab lying
between two spikes). Rule applied here, written to data/raw/complex_overrides.csv for you to review:
  1. every partner chain is assigned to the RBD copy it touches with the most heavy-atom pairs (ties: alphabetically first copy);
  2. the chosen RBD copy is the one whose assigned partners have the largest total contact count;
  3. its antibody chains are the partner chains assigned to it (each with >= --min-pairs pairs).
Chains whose second-best copy has at least --ambiguous (default 0.7) of the best count are listed in the note: look at those first.
Existing rows of the overrides file (same resolved_id and label) are never replaced.

    python scripts/37b_resolve_shared_partners.py && python scripts/37_freeze_complex_map.py --weak-pairs 1 --min-total 40
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
spec = importlib.util.spec_from_file_location("s37", HERE / "37_freeze_complex_map.py")
s37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s37)

COLS = ["resolved_id", "label_contains", "rbd_chain", "antibody_chains", "note"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="data/structures/antibody_rbd/complex_map_final.csv")
    ap.add_argument("--overrides", default="data/raw/complex_overrides.csv")
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--ambiguous", type=float, default=0.7)
    args = ap.parse_args()

    final = pd.read_csv(args.final, dtype=str, keep_default_na=False)
    todo = final[(final["qc_pass"].str.lower() == "false") & final["flags"].str.contains("shared_partner_unresolved")]
    print(f"{len(todo)} rows failed with shared_partner_unresolved")
    old = pd.read_csv(args.overrides, dtype=str, keep_default_na=False) if Path(args.overrides).is_file() else pd.DataFrame(columns=COLS)
    have = set(zip(old["resolved_id"].str.upper(), old["label_contains"]))
    new, review = [], []
    for r in todo.itertuples():
        path = Path(r.file)
        if not path.is_file():
            print(f"  skip {r.label}: no file")
            continue
        _, _, _, per_copy, _, _, _, _ = s37.analyse(path, args.min_pairs)
        if len(per_copy) < 2:
            print(f"  skip {r.label} ({r.resolved_id}): fewer than 2 RBD copies, the flag has another cause")
            continue
        partners = sorted({c for d in per_copy.values() for c in d})
        owner, amb = {}, []
        for c in partners:
            counts = {rc: per_copy[rc].get(c, 0) for rc in sorted(per_copy)}
            best = max(counts.values())
            owner[c] = next(rc for rc in sorted(counts) if counts[rc] == best)
            second = sorted(counts.values(), reverse=True)[1]
            if second > 0 and second >= args.ambiguous * best:
                amb.append(c)
        score = {rc: sum(per_copy[rc][c] for c in partners if owner[c] == rc) for rc in sorted(per_copy)}
        chosen = max(sorted(score), key=lambda rc: score[rc])
        ab = [c for c in sorted(per_copy[chosen], key=lambda c: -per_copy[chosen][c]) if owner[c] == chosen]
        if not ab:
            print(f"  skip {r.label}: no partner assigned to the best copy")
            continue
        review.append({"label": r.label, "id": r.resolved_id, "copies": len(per_copy), "rbd": chosen, "antibody": "+".join(ab),
                       "scores": score, "ambiguous": "".join(amb)})
        if (r.resolved_id.upper(), r.label) in have:
            continue
        new.append({"resolved_id": r.resolved_id, "label_contains": r.label, "rbd_chain": chosen, "antibody_chains": "+".join(ab),
                    "note": f"auto: partners assigned to the RBD copy they touch most; copy scores {score}; ambiguous chains {''.join(amb) or 'none'}"})
    print(pd.DataFrame(review).to_string(index=False))
    out = pd.concat([old, pd.DataFrame(new, columns=COLS)], ignore_index=True)
    Path(args.overrides).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.overrides, index=False)
    print(f"\nwrote {args.overrides}: {len(new)} new rows ({len(old)} kept). Review rows with ambiguous chains, then rerun 37 with --weak-pairs 1 --min-total 40")


if __name__ == "__main__":
    main()
