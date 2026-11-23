#!/usr/bin/env python3
"""Which third field does SCAP want? Builds one mutant with each format and prints the residue found at the site."""
import argparse, importlib.util, shutil, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("m42", HERE / "42_build_scap_mutants.py")
m42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m42)
from topnet.features.pdbio import AA1  # one letter -> three letter


def run(structure, chain, resid, field, scap):
    library = m42.jackal_library(scap)
    with tempfile.TemporaryDirectory(prefix="probe_") as t:
        work = Path(t)
        (work / "complex.pdb").write_text("".join(m42.atom_lines(structure)), encoding="utf-8")
        (work / "tmp_scap.list").write_text(f"{chain},{resid},{field}\n", encoding="utf-8")
        m42.write_jackal_dir(work, library)
        local = work / "scap"
        shutil.copyfile(scap, local)
        local.chmod(0o755)
        proc = subprocess.run(["./scap", "-ini", "20", "-min", "4", "complex.pdb", "tmp_scap.list"],
                              cwd=work, capture_output=True)
        out = next(iter(sorted(work.glob("*_scap.pdb"))), None)
        if proc.returncode != 0 or out is None:
            return f"FAILED exit {proc.returncode}: {m42.text_of(proc.stderr)[-150:]}"
        names = {ln[17:20].strip() for ln in out.read_text(errors="replace").splitlines()
                 if ln.startswith("ATOM") and ln[21] == chain and ln[22:27].strip() == str(resid).strip().upper()}
        return ",".join(sorted(names)) or "site not found"


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
structure = Path(rec["structure"])
if not structure.is_file():
    structure = ROOT / rec["structure"]
target = rec["mut"].upper()
print(f"{rec.get('mutation', '')} {rec['chain']}:{rec['resid_struct']} wanted {target} = {AA1[target]}  ({structure.name})")
for label, field in (("one-letter", target), ("three-letter", AA1[target])):
    print(f"  list field {field!r:6} ({label:12}) -> residue at site: {run(structure, rec['chain'], rec['resid_struct'], field, Path(args.scap))}")