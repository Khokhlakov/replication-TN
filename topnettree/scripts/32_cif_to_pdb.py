#!/usr/bin/env python3
"""Convert a .cif to PDB. Multi-letter chain ids become unused single characters (map written next to the file).
--shift CHAIN:OFFSET adds OFFSET to every residue number of that ORIGINAL chain (use the negative of the step-3 offset)."""
import argparse
import json
import string
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO

ap = argparse.ArgumentParser()
ap.add_argument("cif")
ap.add_argument("--out-dir", default="data/structures/antibody_rbd_converted")
ap.add_argument("--shift", action="append", default=[])
args = ap.parse_args()
cif = Path(args.cif)
model = MMCIFParser(QUIET=True).get_structure(cif.stem, str(cif))[0]
shifts = {k: int(v) for k, v in (x.split(":") for x in args.shift)}
used = {c.id for c in model if len(c.id) == 1}
free = [c for c in string.ascii_uppercase + string.ascii_lowercase + string.digits if c not in used]
mapping = {}
for ch in list(model):
    orig = ch.id
    if orig in shifts:
        for res in list(ch):
            h, n, i = res.id
            res.id = (h, n + 100000, i)
        for res in list(ch):
            h, n, i = res.id
            res.id = (h, n - 100000 + shifts[orig], i)
    if len(orig) > 1:
        ch.id = free.pop(0)
    mapping[orig] = ch.id
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
io = PDBIO()
io.set_structure(model)
io.save(str(out / f"{cif.stem}.pdb"))
(out / f"{cif.stem}.chainmap.json").write_text(json.dumps({"original_to_new": mapping, "shifts": shifts}, indent=1))
print(cif.stem, "->", out / f"{cif.stem}.pdb", mapping)