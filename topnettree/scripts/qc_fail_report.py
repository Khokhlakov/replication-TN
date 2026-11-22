#!/usr/bin/env python3
"""Why did each complex fail QC, and which chains would a rescue use?

Reads data/structures/antibody_rbd/complex_map_final.csv, takes the rows with qc_pass False,
and for each one prints the stored reason (flags, rule, contacts_json) plus a fresh contact
table computed from the structure file: heavy C/N/O atom pairs within --cutoff (4.5 A) between
the chosen RBD chain and every other chain, and for each contacting chain the chain it touches
most among the non-RBD chains (a Fab heavy chain's partner is its light chain).

The suggestion is only a starting point. It cannot know which chains belong to the antibody and
which are spike protomers or a second Fab. Read the table, then decide, as you did for the
hand-set rows in deviations.md.

    python scripts/qc_fail_report.py
    python scripts/qc_fail_report.py --only 7K90 7C01
    python scripts/qc_fail_report.py --all      # also the rows that passed, for comparison

Writes qc_fail_report.csv (one line per complex) next to the map.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from topnet.features.pdbio import read_pdb_records  # noqa: E402


def chain_trees(path):
    recs = [r for r in read_pdb_records(path, altloc="first") if r["element"] in ("C", "N", "O")]
    xyz, res = {}, {}
    for r in recs:
        xyz.setdefault(r["chain"], []).append(r["xyz"])
        res.setdefault(r["chain"], set()).add((r["resseq"], r["icode"]))
    trees = {c: cKDTree(np.asarray(v, dtype=float)) for c, v in xyz.items()}
    return trees, {c: len(v) for c, v in res.items()}, {c: len(v) for c, v in xyz.items()}


def pairs(a, b, cutoff):
    return int(a.count_neighbors(b, cutoff))


def report_row(row, cutoff, min_pairs, partner_pairs):
    path = Path(row["file"])
    if not path.is_absolute():
        path = ROOT / path
    out = {
        "col_index": row["col_index"], "label": row["label"], "id": row["resolved_id"],
        "rbd_chain": row["rbd_chain"], "ab_chains_current": row["ab_chains"],
        "n_rbd_copies": row["n_rbd_copies"], "rule": row["rule"], "flags": row["flags"],
    }
    print(f"\n[{row['col_index']}] {row['label']}  {row['resolved_id']}  qc_pass={row['qc_pass']}")
    print(f"    rbd_chain={row['rbd_chain']}  ab_chains={row['ab_chains']}  rbd_copies={row['n_rbd_copies']}  rule={row['rule']}")
    print(f"    flags: {row['flags']}")
    print(f"    stored contacts: {row['contacts_json']}")
    if not path.is_file():
        print(f"    FILE NOT FOUND: {path}")
        out["suggested"] = ""
        out["note"] = "file not found"
        return out
    trees, n_res, n_atoms = chain_trees(path)
    rbd = row["rbd_chain"]
    if rbd not in trees:
        print(f"    chosen RBD chain {rbd!r} is not in the file (chains: {''.join(sorted(trees))})")
        out["suggested"] = ""
        out["note"] = "rbd chain not in file"
        return out
    current = set(row["ab_chains"].split("+")) if row["ab_chains"] else set()
    touch = {c: pairs(trees[rbd], t, cutoff) for c, t in trees.items() if c != rbd}
    print(f"    chains in file: {' '.join(sorted(trees))}   (RBD chain {rbd}: {n_res[rbd]} residues)")
    print(f"    {'chain':>5} {'res':>5} {'pairs->RBD':>10}  {'best partner (pairs)':<22} in current")
    candidates = []
    for c in sorted(touch, key=lambda k: -touch[k]):
        if touch[c] == 0 and c not in current:
            continue
        others = {d: pairs(trees[c], trees[d], cutoff) for d in trees if d not in (c, rbd)}
        best = max(others, key=others.get) if others else None
        best_txt = f"{best} ({others[best]})" if best and others[best] else "-"
        print(f"    {c:>5} {n_res[c]:>5} {touch[c]:>10}  {best_txt:<22} {'yes' if c in current else ''}")
        if touch[c] >= min_pairs:
            if c not in candidates:
                candidates.append(c)
            if best and others[best] >= partner_pairs and best not in candidates:
                candidates.append(best)
    out["suggested"] = "+".join(candidates)
    note = []
    try:
        if int(row["n_rbd_copies"]) > 1:
            note.append("several RBD copies: choose the copy and keep only its Fab by hand")
    except ValueError:
        pass
    if set(candidates) == current:
        note.append("suggestion equals the current assignment: the failure is a flag, not the chains")
    out["note"] = "; ".join(note)
    print(f"    suggested ab_chains (review): {out['suggested'] or '-'}" + (f"   <- {out['note']}" if note else ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="data/structures/antibody_rbd/complex_map_final.csv")
    ap.add_argument("--cutoff", type=float, default=4.5)
    ap.add_argument("--min-pairs", type=int, default=5, help="contacts for a chain to be a candidate")
    ap.add_argument("--partner-pairs", type=int, default=100, help="contacts that make two chains one Fab")
    ap.add_argument("--only", nargs="*", help="resolved ids to report")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    frame = pd.read_csv(args.map, dtype=str, keep_default_na=False)
    mask = frame["qc_pass"].str.lower() != "true"
    sel = frame if args.all else frame[mask]
    if args.only:
        sel = sel[sel["resolved_id"].isin(args.only)]
    print(f"{int(mask.sum())} of {len(frame)} rows fail QC; reporting {len(sel)}")
    rows = [report_row(r, args.cutoff, args.min_pairs, args.partner_pairs) for r in sel.to_dict("records")]
    out = Path(args.map).with_name("qc_fail_report.csv")
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
