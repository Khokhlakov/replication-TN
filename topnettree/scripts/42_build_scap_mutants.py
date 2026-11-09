#!/usr/bin/env python3
"""Build mutant PDBs with the capsule SCAP binary.

One list line, ``chain,resid,three-letter``, then ``scap -ini 20 -min 4``.
The list name has to contain ``_scap.list``; that is what scap.cpp accepts.
profix runs only when --profix is passed. A missing binary or a failed row is
recorded and no coordinate file is invented for that row.

SCAP opens ``./jackal.dir`` and then the library path written in that file.
The capsule copy names ``/home/phantom/Downloads/software/jackal_64bit``, which
is not this machine. Each job gets its own ``jackal.dir`` in the work
directory, with ``library:`` set to the capsule's ``jackal_64bit`` directory.
The capsule file is not edited. The binary is copied into the work directory
and marked executable, because the shipped file is often not executable and
``ldd`` only reports that it is statically linked.

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


def text_of(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", "replace").replace("\n", " | ")[-500:]


def jackal_root(scap: Path) -> Path:
    root = scap.resolve().parent.parent
    if not (root / "library" / "tres").is_file():
        raise RuntimeError(f"{root} has no library/tres; scap is not inside jackal_64bit/bin")
    return root


def write_jackal_dir(work: Path, library: Path) -> None:
    (work / "jackal.dir").write_text(
        f"pdb: .\nlibrary: {library}\nkey:XIANG2002\n",
        encoding="utf-8",
    )


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
        library = jackal_root(scap)
        (work / "complex.pdb").write_text("".join(atom_lines(source)), encoding="utf-8")
        (work / "tmp_scap.list").write_text(
            f"{rec['chain']},{rec['resid_struct']},{AA1[rec['mut']]}\n", encoding="utf-8"
        )
        write_jackal_dir(work, library)
        local_scap = work / "scap"
        shutil.copyfile(scap, local_scap)
        local_scap.chmod(0o755)
        proc = subprocess.run(
            ["./scap", "-ini", "20", "-min", "4", "complex.pdb", "tmp_scap.list"],
            cwd=work, capture_output=True,
        )
        produced = next(iter(sorted(work.glob("*_scap.pdb"))), None)
        detail = text_of(proc.stderr) or text_of(proc.stdout)
        if proc.returncode != 0 or produced is None:
            return {"mutant": rec["mutant"], "status": f"scap_exit_{proc.returncode}", "stderr": detail}
        if profix is not None:
            shutil.copyfile(produced, work / "complex.pdb")
            local_profix = work / "profix"
            shutil.copyfile(profix, local_profix)
            local_profix.chmod(0o755)
            repair = subprocess.run(["./profix", "-fix", "0", "complex.pdb"], cwd=work, capture_output=True)
            fixed = work / "complex_fix.pdb"
            if repair.returncode != 0 or not fixed.is_file():
                return {"mutant": rec["mutant"], "status": f"profix_exit_{repair.returncode}",
                        "stderr": text_of(repair.stderr) or text_of(repair.stdout)}
            produced = fixed
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, dest)
        return {"mutant": rec["mutant"], "status": "ok", "stderr": ""}
    except Exception as err:
        return {"mutant": rec["mutant"], "status": "error", "stderr": f"{type(err).__name__}: {err}"[:500]}
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
    failed = [item for item in log if item["status"] not in ("ok", "cached")]
    if failed:
        print(failed[0]["mutant"])
        print(failed[0].get("stderr", ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
