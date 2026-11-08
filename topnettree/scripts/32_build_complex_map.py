#!/usr/bin/env python3
"""Step 3a. Build data/structures/antibody_rbd/complex_map.csv, the input of 37_freeze_complex_map.py, from the manifest
that 30_fetch_antibody_structures.py wrote. 37 recomputes chains and contacts itself and only needs four columns here:
col_index, label, resolved_id, file.

PATCH: when the manifest file is a .cif and data/structures/antibody_rbd_converted/<ID>.pdb exists, that converted PDB is used.
The converted files carry the chain renaming and the residue shifts (for example 7B27) applied by 32_cif_to_pdb.py; reading the
original .cif would silently bring the unshifted numbering back.

    python scripts/32_build_complex_map.py [--manifest data/structures/antibody_rbd/manifest.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/structures/antibody_rbd/manifest.csv")
ap.add_argument("--converted-dir", default="data/structures/antibody_rbd_converted")
ap.add_argument("--out", default="data/structures/antibody_rbd/complex_map.csv")
args = ap.parse_args()
man = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)


def pick(row) -> str:
    p = row["path"]
    if p.lower().endswith(".cif"):
        conv = Path(args.converted_dir) / f"{row['resolved_id']}.pdb"
        if conv.is_file():
            return str(conv)
    return p


cmap = pd.DataFrame({"col_index": man["col_index"], "label": man["label"], "resolved_id": man["resolved_id"],
                     "file": man.apply(pick, axis=1)})
cif_rows = cmap[cmap["file"].str.lower().str.endswith(".cif")]
cmap.to_csv(args.out, index=False)
converted = man[man["path"].str.lower().str.endswith(".cif")]
print(f"wrote {args.out}: {len(cmap)} rows; .cif entries in the manifest: {len(converted)}; "
      f"still pointing at a .cif (no converted PDB found): {len(cif_rows)}")
print(converted[["col_index", "label", "resolved_id"]].to_string(index=False))
missing = cmap[~cmap["file"].map(lambda p: bool(p) and Path(p).is_file())]
print(f"rows whose file is missing: {len(missing)}")
if len(missing):
    print(missing.to_string(index=False))
