#!/usr/bin/env python3
"""Manifests for build_homology_features.py from the tables you already have.

  panel   data/predict/rows_26x186.csv  (26 mutations x 186 complexes)
          chain       = rbd_chain (the RBD carries the mutation)
          keep_chains = rbd_chain + ab_chains, e.g. E+B+C+D+A
          mutant_pdb  = --mutant-template filled from the row (default
                        <id>_<chain>_<wt>_<resid>_<mut>.pdb), OR, with --panel-rows, the
                        `mutant` path and `resid_struct` from data/features/rows/panel.csv.
                        Use --panel-rows: 42_build_scap_mutants.py builds exactly the files
                        that panel.csv names, and numbers residues by resid_struct.
          Rows are dropped when qc_pass is not True (unless --include-qc-fail),
          site_present is 0 or wt_ok is 0.
          row_id = pair_id, which already contains the complex column index.

  dms     data/features/rows/starr2020.csv (or chan2020.csv, same columns)
          wild_pdb = structure, mutant_pdb = mutant, chain = chain, resid = resid_struct
          keep_chains = --keep-chains (E+A for 6M0J, whichever chain is mutated).
          Rows with wt_ok != 1 are dropped.

Every drop is counted and printed. Missing mutant files are counted but kept in the
manifest, because `build_homology_features.py check` is the step that reports them.

    python scripts/make_manifests.py panel --rows data/predict/rows_26x186.csv \
        --out data/features/panel_manifest.csv
    python scripts/make_manifests.py dms --rows data/features/rows/starr2020.csv \
        --keep-chains E+A --out data/features/starr_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_TEMPLATE = "data/features/mutants/{resolved_id}_{rbd_chain}_{wt}_{resid}_{mut}.pdb"


def finish(frame, out, label):
    cols = ["row_id", "wild_pdb", "mutant_pdb", "chain", "resid", "wt", "mut", "keep_chains"]
    frame = frame[cols]
    dup = frame["row_id"][frame["row_id"].duplicated()].tolist()
    if dup:
        raise SystemExit(f"duplicate row_id values: {dup[:5]}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, lineterminator="\n")
    have_wild = int(frame["wild_pdb"].map(lambda p: Path(p).is_file()).sum())
    have_mut = int(frame["mutant_pdb"].map(lambda p: Path(p).is_file()).sum())
    print(f"{label}: wrote {len(frame)} rows to {out}")
    print(f"  wild files present   {have_wild}/{len(frame)}")
    print(f"  mutant files present {have_mut}/{len(frame)}")
    if have_mut < len(frame):
        print("  Missing mutants have to be built (42_build_scap_mutants.py) before `build` can run on those rows.")


def join_panel_rows(keep, panel_path):
    """Attach resid_struct and the mutant path from panel.csv; drop and report anything inconsistent."""
    pr = pd.read_csv(panel_path, dtype=str, keep_default_na=False)
    need = ["mutation", "label", "pdb", "chain", "wt", "resid_struct", "mut", "structure", "mutant"]
    missing = [c for c in need if c not in pr.columns]
    if missing:
        raise SystemExit(f"{panel_path} lacks columns {missing}")
    if pr.duplicated(["label", "mutation"]).any():
        raise SystemExit(f"{panel_path} has several rows for the same (label, mutation); cannot join")
    pr = pr[need].rename(columns={c: "p_" + c for c in need if c not in ("label", "mutation")})
    merged = keep.merge(pr, on=["label", "mutation"], how="left")
    unmatched = merged["p_mutant"].isna()
    bad = (
        (merged["p_pdb"] != merged["resolved_id"])
        | (merged["p_chain"] != merged["rbd_chain"])
        | (merged["p_wt"] != merged["wt"])
        | (merged["p_mut"] != merged["mut"])
        | (merged["p_structure"] != merged["file"])
    ) & ~unmatched
    print(f"join with {panel_path}: {int(unmatched.sum())} rows not in panel.csv, {int(bad.sum())} rows disagree on pdb/chain/wt/mut/structure")
    for _, r in merged[bad].head(5).iterrows():
        print(f"  disagree {r['pair_id']}: panel.csv says {r['p_pdb']}/{r['p_chain']}/{r['p_wt']}->{r['p_mut']} {r['p_structure']}")
    return merged[~unmatched & ~bad].copy()


def panel(args):
    raw = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
    need = ["pair_id", "resolved_id", "file", "rbd_chain", "ab_chains", "qc_pass", "wt", "resid", "mut", "site_present", "wt_ok"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit(f"{args.rows} lacks columns {missing}")
    n0 = len(raw)
    qc = raw["qc_pass"].str.lower() == "true"
    present = raw["site_present"] == "1"
    wt_ok = raw["wt_ok"] == "1"
    print(f"panel rows {n0}: qc_pass false {int((~qc).sum())}, site absent {int((~present).sum())}, wild-type letter mismatch {int((~wt_ok).sum())}")
    keep = raw[(qc | args.include_qc_fail) & present & wt_ok].copy()
    print(f"kept {len(keep)}, dropped {n0 - len(keep)} (a row can fail more than one test)")
    keep["keep_chains"] = keep["rbd_chain"] + "+" + keep["ab_chains"]
    # The mutated chain must be one character, and keep_chains must contain it once.
    bad = keep[keep["rbd_chain"].str.len() != 1]
    if len(bad):
        raise SystemExit(f"rbd_chain is not one character in rows {bad['pair_id'].tolist()[:5]}")
    if args.panel_rows:
        keep = join_panel_rows(keep, args.panel_rows)
        keep["mutant_pdb"] = keep["p_mutant"]
        keep["resid_out"] = keep["p_resid_struct"]
        differ = int((keep["p_resid_struct"] != keep["resid"]).sum())
        print(f"resid_struct differs from the paper resid in {differ} kept rows (the structure number is used)")
    else:
        keep["mutant_pdb"] = [
            args.mutant_template.format(resolved_id=r.resolved_id, rbd_chain=r.rbd_chain, wt=r.wt, resid=r.resid, mut=r.mut)
            for r in keep.itertuples()
        ]
        keep["resid_out"] = keep["resid"]
    frame = pd.DataFrame(
        {
            "row_id": keep["pair_id"],
            "wild_pdb": keep["file"],
            "mutant_pdb": keep["mutant_pdb"],
            "chain": keep["rbd_chain"],
            "resid": keep["resid_out"],
            "wt": keep["wt"],
            "mut": keep["mut"],
            "keep_chains": keep["keep_chains"],
        }
    )
    finish(frame, args.out, "panel")


def dms(args):
    raw = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
    need = ["row_idx", "dataset", "chain", "wt", "resid_struct", "mut", "structure", "mutant", "wt_ok"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit(f"{args.rows} lacks columns {missing}")
    ok = raw["wt_ok"] == "1"
    print(f"dms rows {len(raw)}: wt_ok != 1 for {int((~ok).sum())}")
    keep = raw[ok]
    chains = set(args.keep_chains.split("+"))
    off = keep[~keep["chain"].isin(chains)]
    if len(off):
        raise SystemExit(f"{len(off)} rows have a mutated chain outside --keep-chains {args.keep_chains}")
    frame = pd.DataFrame(
        {
            "row_id": keep["dataset"] + "_" + keep["row_idx"],
            "wild_pdb": keep["structure"],
            "mutant_pdb": keep["mutant"],
            "chain": keep["chain"],
            "resid": keep["resid_struct"],
            "wt": keep["wt"],
            "mut": keep["mut"],
            "keep_chains": args.keep_chains,
        }
    )
    finish(frame, args.out, "dms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("panel")
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mutant-template", default=DEFAULT_TEMPLATE)
    p.add_argument("--panel-rows", help="data/features/rows/panel.csv; recommended")
    p.add_argument("--include-qc-fail", action="store_true", help="keep complexes with qc_pass False (for diagnosis)")
    p.set_defaults(func=panel)
    p = sub.add_parser("dms")
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep-chains", required=True, help="e.g. E+A for 6M0J")
    p.set_defaults(func=dms)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
