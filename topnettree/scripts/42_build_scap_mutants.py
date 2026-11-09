#!/usr/bin/env python3
"""Build mutant PDBs with the capsule SCAP binary.

One list line, ``chain,resid,three-letter``, then ``scap -ini 20 -min 4``.
profix runs only when --profix is passed. A missing binary or a failed row is
recorded and no coordinate file is invented for that row.

The residue number is resid_struct: the number in the structure file named by
the ``structure`` column.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from topnet.features.pdbio import AA1  # noqa: E402


def atom_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        if line.startswith("ATOM"):
            lines.append(line if line.endswith("\n") else line + "\n")
    if not lines:
        raise RuntimeError(f"no ATOM records in {path}")
    return lines


def build_one(rec: dict, scap: Path, profix: Path | None, work_root: Path) -> dict:
    dest = ROOT / rec["mutant"]
    if dest.is_file():
        return {"mutant": rec["mutant"], "status": "cached"}
    source = Path(rec["structure"])
    if not source.is_file():
        source = ROOT / rec["structure"]
    if not source.is_file():
        return {"mutant": rec["mutant"], "status": "missing_structure", "stderr": rec["structure"]}
    work = work_root / dest.stem
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        (work / "complex.pdb").write_text("".join(atom_lines(source)), encoding="utf-8")
        (work / "tmp_scap.list").write_text(
            f"{rec['chain']},{rec['resid_struct']},{AA1[rec['mut']]}\n", encoding="utf-8"
        )
        proc = subprocess.run(
            [str(scap), "-ini", "20", "-min", "4", "complex.pdb", "tmp_scap.list"],
            cwd=work, capture_output=True, text=True,
        )
        produced = work / "complex_scap.pdb"
        if proc.returncode != 0 or not produced.is_file():
            return {"mutant": rec["mutant"], "status": f"scap_exit_{proc.returncode}",
                    "stderr": (proc.stderr or proc.stdout)[-300:].replace("\n", " | ")}
        if profix is not None:
            shutil.copyfile(produced, work / "complex.pdb")
            repair = subprocess.run([str(profix), "-fix", "0", str(work / "complex.pdb")],
                                    cwd=work, capture_output=True, text=True)
            fixed = work / "complex_fix.pdb"
            if repair.returncode != 0 or not fixed.is_file():
                return {"mutant": rec["mutant"], "status": f"profix_exit_{repair.returncode}",
                        "stderr": (repair.stderr or repair.stdout)[-300:].replace("\n", " | ")}
            produced = fixed
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, dest)
        return {"mutant": rec["mutant"], "status": "ok"}
    except Exception as err:
        return {"mutant": rec["mutant"], "status": "error", "stderr": str(err)[:300]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--scap", required=True)
    parser.add_argument("--profix")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--work", default="data/features/mutants/_work")
    args = parser.parse_args()
    scap = Path(args.scap)
    if not scap.is_file():
        raise SystemExit(f"SCAP binary not found: {scap}")
    profix = Path(args.profix) if args.profix else None
    if profix is not None and not profix.is_file():
        raise SystemExit(f"profix binary not found: {profix}")
    rows = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
    if args.limit:
        rows = rows.iloc[: args.limit]
    work = ROOT / args.work
    work.mkdir(parents=True, exist_ok=True)
    log = [build_one(rec, scap, profix, work) for rec in rows.to_dict("records")]
    out = Path(args.rows).with_suffix(".scap_log.csv")
    pd.DataFrame(log).to_csv(out, index=False)
    counts = pd.Series([item["status"] for item in log]).value_counts().to_dict()
    print(counts)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
