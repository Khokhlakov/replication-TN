#!/usr/bin/env python3
"""Step 3e. Propose antibody_chains for flagged complexes using antibody UNITS (a Fab = heavy + light, a nanobody = one chain).

Why: 37b assigned each chain to an RBD copy one chain at a time, which can split a Fab (a light chain that barely touches the
RBD is assigned to the wrong copy or dropped) and 37 can also take spike chains as 'partners'. Here:
  1. candidates: chains whose RCSB description looks like an antibody (and not spike/RBD/ACE2); chains without a description
     only if they touch an RBD copy with >= --nodesc-pairs pairs;
  2. units: heavy and light chains are paired one-to-one by their mutual contact (strongest first, >= --unit-pairs pairs), so two
     neighbouring Fabs are never merged; nanobodies and unpaired chains are single-chain units;
  3. each unit goes to the RBD copy it touches most (pairs summed over its chains), kept if that total >= --min-unit;
  4. chosen RBD copy: largest total of its units, but the current copy is kept if within --keep-copy (0.95) of the best;
  5. antibody_chains: every chain of those units that touches the chosen copy with >= 1 pair (QC then needs --weak-pairs 1).
Rows whose label contains ACE2 or CTC are skipped. Hand-written override rows (note not starting with 'auto') are never
replaced unless --force.

    python scripts/37d_propose_by_units.py                 # print proposals for flagged rows (changes only)
    python scripts/37d_propose_by_units.py --write         # update data/raw/complex_overrides.csv
    python scripts/37d_propose_by_units.py --labels 7K8X 7C01
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s37 = load("s37", "37_freeze_complex_map.py")
r37c = load("r37c", "37c_review_chains.py")
c31 = s37.c31
COLS = ["resolved_id", "label_contains", "rbd_chain", "antibody_chains", "note"]


NANO = re.compile(r"nanobod|nonobody|\bvhh\b|sybody|single.domain|\bnb\d*\b|binder|\bvh\b", re.I)


def role(desc: str) -> str:
    if not desc:
        return "?"
    if NANO.search(desc):
        return "N"
    if r37c.HEAVY.search(desc) and not r37c.LIGHT.search(desc):
        return "H"
    if r37c.LIGHT.search(desc) and not r37c.HEAVY.search(desc):
        return "L"
    return "?"


def pair_units(cand, desc, contact_fn, min_pairs):
    """Fab = one heavy + one light chain, paired greedily by their mutual contact count (strongest pair first), so two
    neighbouring Fabs are never merged. Chains described only as '<name> antibody' pair with another chain of the same
    description. Nanobodies, undescribed chains and leftovers are single-chain units."""
    eligible = []
    for a, b in itertools.combinations(cand, 2):
        ra, rb = role(desc.get(a, "")), role(desc.get(b, ""))
        if {ra, rb} == {"H", "L"} or (ra == rb == "?" and desc.get(a) and desc.get(a) == desc.get(b)):
            eligible.append((contact_fn(a, b), a, b))
    used, units = set(), []
    for n, a, b in sorted(eligible, reverse=True):
        if n >= min_pairs and a not in used and b not in used:
            units.append(sorted([a, b]))
            used |= {a, b}
    units += [[c] for c in cand if c not in used]
    return units


def components(chains, linked):
    parent = {c: c for c in chains}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in linked:
        parent[find(a)] = find(b)
    groups = {}
    for c in chains:
        groups.setdefault(find(c), []).append(c)
    return [sorted(g) for g in groups.values()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="data/structures/antibody_rbd/complex_map_final.csv")
    ap.add_argument("--review", default="data/structures/antibody_rbd/complex_chain_review.csv")
    ap.add_argument("--cache", default="data/raw/rcsb_entities")
    ap.add_argument("--overrides", default="data/raw/complex_overrides.csv")
    ap.add_argument("--labels", nargs="*", help="PDB ids or labels to process instead of the flagged rows")
    ap.add_argument("--unit-pairs", type=int, default=100)
    ap.add_argument("--min-unit", type=int, default=20)
    ap.add_argument("--nodesc-pairs", type=int, default=20)
    ap.add_argument("--keep-copy", type=float, default=0.95)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    final = pd.read_csv(args.final, dtype=str, keep_default_na=False)
    if args.labels:
        want = {x.lower() for x in args.labels}
        todo = final[final["resolved_id"].str.lower().isin(want) | final["label"].str.lower().isin(want)]
    else:
        rev = pd.read_csv(args.review, dtype=str, keep_default_na=False)
        todo = final[final["col_index"].isin(rev[rev["flags"] != ""]["col_index"])]
    old = pd.read_csv(args.overrides, dtype=str, keep_default_na=False) if Path(args.overrides).is_file() else pd.DataFrame(columns=COLS)
    out_rows, shown = [], []
    for r in todo.itertuples():
        lab = r.label.lower()
        if "ace2" in lab or "ctc" in lab or not r.file or not Path(r.file).is_file():
            continue
        atoms, _, _, per_copy, _, _, _, _ = s37.analyse(Path(r.file), 1)
        if not per_copy:
            continue
        desc = r37c.entity_chains(r.resolved_id.upper(), Path(args.cache))
        touching = {c for d in per_copy.values() for c in d}
        cand = []
        for c in sorted(touching):
            d = desc.get(c, "")
            if d and r37c.ABLIKE.search(d) and not r37c.is_spike(d):
                cand.append(c)
            elif not d and max(per_copy[rc].get(c, 0) for rc in per_copy) >= args.nodesc_pairs:
                cand.append(c)
        units = pair_units(cand, desc, lambda a, b: c31.contacts(atoms[a], atoms[b], 4.5)[0], args.unit_pairs)
        owned = {rc: [] for rc in per_copy}
        text = []
        for u in units:
            tot = {rc: sum(per_copy[rc].get(c, 0) for c in u) for rc in sorted(per_copy)}
            owner = max(sorted(tot), key=lambda k: tot[k])
            text.append(f"{'+'.join(u)}->{owner}:{tot[owner]}")
            if tot[owner] >= args.min_unit:
                owned[owner].append((u, tot[owner]))
        score = {rc: sum(t for _, t in v) for rc, v in owned.items()}
        best = max(sorted(score), key=lambda k: score[k])
        chosen = r.rbd_chain if r.rbd_chain in score and score[r.rbd_chain] >= args.keep_copy * score[best] else best
        ab = []
        for u, _ in owned[chosen]:
            ab += [c for c in u if per_copy[chosen].get(c, 0) >= 1]
        ab = sorted(set(ab), key=lambda c: -per_copy[chosen].get(c, 0))
        cur = (r.rbd_chain, sorted(r.ab_chains.split("+")))
        new = (chosen, sorted(ab))
        if not ab or cur == new:
            continue
        shown.append({"label": r.label, "id": r.resolved_id, "current": f"{cur[0]}:{'+'.join(cur[1])}",
                      "proposed": f"{chosen}:{'+'.join(ab)}", "units": " ".join(text)})
        out_rows.append({"resolved_id": r.resolved_id, "label_contains": r.label, "rbd_chain": chosen, "antibody_chains": "+".join(ab),
                         "note": f"auto-units: {' '.join(text)}"})
    if not shown:
        print("no changes proposed")
        return
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        print(pd.DataFrame(shown).to_string(index=False))
    if args.write:
        keep = old.copy()
        for row in out_rows:
            hit = (keep["resolved_id"].str.upper() == row["resolved_id"].upper()) & (keep["label_contains"] == row["label_contains"])
            if hit.any():
                if not args.force and not keep.loc[hit, "note"].str.startswith("auto").all():
                    print(f"  kept hand-written row for {row['label_contains']} ({row['resolved_id']}); use --force to replace")
                    continue
                keep = keep[~hit]
            keep = pd.concat([keep, pd.DataFrame([row], columns=COLS)], ignore_index=True)
        keep.to_csv(args.overrides, index=False)
        print(f"wrote {args.overrides} ({len(keep)} rows). Rerun: python scripts/37_freeze_complex_map.py --weak-pairs 1 --min-total 40")


if __name__ == "__main__":
    main()
