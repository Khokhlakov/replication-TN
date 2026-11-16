#!/usr/bin/env python3
"""Manifest for build_homology_features.py from a folder of mutant PDBs.

File names are <PDB>_<chain>_<wt>_<resid>_<mut>.pdb (for example 1A4Y_B_H_13_A.pdb).
The wild-type file is <pdb-dir>/<PDB>.pdb. keep_chains is left blank, which keeps the
capsule behaviour of using every chain in the file.

    python scripts/make_manifest_from_mutants.py \
        --mutants-dir data/features/mutants --pdb-dir data/raw/skempi/pdbs \
        --out data/features/s1131_manifest.csv [--limit 20]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutants-dir", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows, skipped = [], []
    for path in sorted(Path(args.mutants_dir).glob("*.pdb")):
        parts = path.stem.split("_")
        if len(parts) != 5 or len(parts[1]) != 1 or len(parts[2]) != 1 or len(parts[4]) != 1:
            skipped.append((path.name, "name is not PDB_chain_wt_resid_mut"))
            continue
        pdb, chain, wt, resid, mut = parts
        wild = Path(args.pdb_dir) / f"{pdb}.pdb"
        if not wild.is_file():
            skipped.append((path.name, f"no wild-type file {wild}"))
            continue
        rows.append(
            {
                "row_id": path.stem,
                "wild_pdb": str(wild),
                "mutant_pdb": str(path),
                "chain": chain,
                "resid": resid,
                "wt": wt,
                "mut": mut,
                "keep_chains": "",
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["row_id"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}; skipped {len(skipped)}")
    for name, why in skipped[:10]:
        print(f"  skipped {name}: {why}")


if __name__ == "__main__":
    main()
