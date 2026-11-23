#!/usr/bin/env python3
"""What does SCAP do with an insertion-coded site? Prints the residues around the site in SCAP's output
for several spellings of the residue field.

    python scripts/scap_site_probe.py --rows data/raw/s645_for_features.csv --scap $CAP/data/software/jackal_64bit/bin/scap \
        --where pdb=1N8Z --where chain=H --where resid=100a --where mut=F
"""
import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("m42", HERE / "42_build_scap_mutants.py")
m42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m42)


def around(text, chain, number, width=2):
    seen = {}
    for ln in text.splitlines():
        if ln.startswith("ATOM") and ln[21] == chain:
            try:
                n = int(ln[22:26])
            except ValueError:
                continue
            if abs(n - number) <= width:
                seen.setdefault((n, ln[26]), ln[17:20])
    return "  ".join(f"{n}{ic.strip() or '_'}:{r}" for (n, ic), r in seen.items()) or "(nothing)"


ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--scap", required=True)
ap.add_argument("--where", action="append", default=[], metavar="COL=VAL")
args = ap.parse_args()
rows = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
for cond in args.where:
    col, val = cond.split("=", 1)
    rows = rows[rows[col] == val]
if rows.empty:
    sys.exit("no row matches")
rec = rows.iloc[0]
src = Path(rec["structure"])
if not src.is_file():
    src = ROOT / rec["structure"]
scap = Path(args.scap)
number = int(re.match(r"-?\d+", rec["resid_struct"]).group())
text_in = "".join(m42.atom_lines(src))
print(f"{rec['pdb']} {rec['chain']}:{rec['resid_struct']} -> {rec['mut']}")
print("input :", around(text_in, rec["chain"], number))
for v in dict.fromkeys([rec["resid_struct"], rec["resid_struct"].upper(), str(number)]):
    with tempfile.TemporaryDirectory(prefix="probe_") as t:
        work = Path(t)
        (work / "complex.pdb").write_text(text_in, encoding="utf-8")
        (work / "tmp_scap.list").write_text(f"{rec['chain']},{v},{rec['mut'].upper()}\n", encoding="utf-8")
        m42.write_jackal_dir(work, m42.jackal_library(scap))
        local = work / "scap"
        shutil.copyfile(scap, local)
        local.chmod(0o755)
        proc = subprocess.run(["./scap", "-ini", "20", "-min", "4", "complex.pdb", "tmp_scap.list"], cwd=work, capture_output=True)
        out = next(iter(sorted(work.glob("*_scap.pdb"))), None)
        print(f"\nlist resid {v!r}: exit {proc.returncode}, output file {'yes' if out else 'NO'}")
        print("   log tail:", (m42.text_of(proc.stderr) or m42.text_of(proc.stdout))[-200:])
        if out:
            print("   output:", around(out.read_text(errors="replace"), rec["chain"], number))