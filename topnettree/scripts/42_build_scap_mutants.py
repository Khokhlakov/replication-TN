#!/usr/bin/env python3
"""Build mutant PDBs with the capsule SCAP binary.

One list line, ``chain,resid,one-letter-code``, then ``scap -ini 20 -min 4``.
The list name has to contain ``_scap.list``; that is what scap.cpp accepts.
profix runs only when --profix is passed. A missing binary or a failed row is
recorded and no coordinate file is invented for that row.

SCAP opens ``./jackal.dir`` and then ``<library>/tres``. ``tres`` lives in
``jackal_64bit/library``, not in ``jackal_64bit`` itself. The capsule copy of
``jackal.dir`` names ``/home/phantom/Downloads/software/jackal_64bit``, which
is not this machine. Each job gets its own ``jackal.dir`` in the work
directory, with ``library:`` set to the absolute ``jackal_64bit/library`` path.
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
import re
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


def jackal_library(scap: Path) -> Path:
    """Directory SCAP searches as ``<library>/tres``."""
    library = scap.resolve().parent.parent / "library"
    if not (library / "tres").is_file():
        raise RuntimeError(f"{library} has no tres; scap is not inside jackal_64bit/bin")
    return library


def write_jackal_dir(work: Path, library: Path) -> None:
    (work / "jackal.dir").write_text(
        f"pdb: .\nlibrary: {library}\nkey:XIANG2002\n",
        encoding="utf-8",
    )


def usable_mutant(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    with path.open("rb") as handle:
        return b"ATOM" in handle.read(4000)


_RESID = re.compile(r"^(-?\d+)([A-Za-z]?)$")


def _residue_key(line):
    return (line[21], line[22:26].strip(), line[26].strip().upper())


def renumber_for_scap(lines, chain, resid):
    """SCAP renumbers chains that carry insertion codes and then reads the site number in its own numbering.
    Give such chains plain consecutive numbers here. Returns (lines, site_number, back); back lists the
    original (number, icode) per residue, by position, for the renumbered chains."""
    m = _RESID.match(str(resid).strip())
    if not m:
        raise ValueError(f"cannot parse residue number {resid!r}")
    want = (m.group(1), m.group(2).upper())
    order, last = {}, None
    for ln in lines:
        k = _residue_key(ln)
        if k != last:
            order.setdefault(k[0], []).append((k[1], k[2]))
            last = k
    renum = {c for c, res in order.items() if any(ic for _, ic in res)}
    out, idx, last, site_no = [], {}, None, None
    for ln in lines:
        k = _residue_key(ln)
        c = k[0]
        if k != last:
            idx[c] = idx.get(c, -1) + 1
            last = k
        if c in renum:
            n = int(order[c][0][0]) + idx[c]
            if not -999 <= n <= 9999:
                raise ValueError(f"renumbered residue {n} does not fit the PDB column")
            ln = ln[:22] + f"{n:>4}" + " " + ln[27:]
            if c == chain and (k[1], k[2]) == want and site_no is None:
                site_no = n
        out.append(ln)
    if chain in renum:
        if site_no is None:
            raise ValueError(f"site {resid} not found in chain {chain}")
    elif want in order.get(chain, []):
        site_no = int(want[0])
    else:
        raise ValueError(f"site {resid} not found in chain {chain}")
    return out, site_no, {c: order[c] for c in renum}


def restore_numbering(text, back):
    """Put the original residue numbers and insertion codes back by position (SCAP keeps residue order)."""
    if not back:
        return text
    out, idx, last = [], {}, None
    for ln in text.splitlines(keepends=True):
        if ln.startswith("ATOM") and len(ln) > 26 and ln[21] in back:
            k = _residue_key(ln)
            if k != last:
                idx[k[0]] = idx.get(k[0], -1) + 1
                last = k
            if idx[k[0]] >= len(back[k[0]]):
                raise ValueError(f"chain {k[0]} has more residues than the input")
            n, ic = back[k[0]][idx[k[0]]]
            ln = ln[:22] + f"{int(n):>4}" + (ic or " ") + ln[27:]
        out.append(ln)
    return "".join(out)


def residue_order(lines):
    out, last = {}, None
    for ln in lines:
        if ln.startswith("ATOM") and len(ln) > 26:
            k = _residue_key(ln)
            if k != last:
                out.setdefault(k[0], []).append((k[1], k[2], ln[17:20].strip()))
                last = k
    return out


def check_site(src_lines, out_text, chain, resid, want3):
    """'' if the output differs from the input at exactly the requested residue (matched by position)."""
    a, b = residue_order(src_lines), residue_order(out_text.splitlines())
    if set(a) != set(b) or any(len(a[c]) != len(b[c]) for c in a):
        return "chain or residue counts differ between input and SCAP output"
    m = _RESID.match(str(resid).strip())
    want = (m.group(1), m.group(2).upper())
    pos = next((i for i, (n, ic, _) in enumerate(a.get(chain, [])) if (n, ic) == want), None)
    if pos is None:
        return f"site {resid} not in input chain {chain}"
    diff = [(c, i) for c in a for i, (x, y) in enumerate(zip(a[c], b[c])) if x[2] != y[2]]
    if not diff:
        return "" if a[chain][pos][2] == want3 else "no residue changed"
    if diff != [(chain, pos)]:
        return "changed " + ", ".join(f"{c}:{a[c][i][0]}{a[c][i][1]} {a[c][i][2]}->{b[c][i][2]}" for c, i in diff[:4])
    return "" if b[chain][pos][2] == want3 else f"built {b[chain][pos][2]}, wanted {want3}"


def build_one(rec: dict, scap: Path, profix: Path | None, work_root: Path, force: bool) -> dict:
    dest = ROOT / rec["mutant"]
    if not force and usable_mutant(dest):
        return {"mutant": str(dest), "status": "cached", "stderr": f"{dest.stat().st_size} bytes"}
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
        library = jackal_library(scap)
        
        target = rec["mut"].strip().upper()
        if target not in AA1:
            raise ValueError(f"mut {rec['mut']!r} is not a one-letter amino acid")
        
        src_lines = atom_lines(source)
        lines, site_no, back = renumber_for_scap(src_lines, rec["chain"], rec["resid_struct"])
        (work / "complex.pdb").write_text("".join(lines), encoding="utf-8")
        (work / "tmp_scap.list").write_text(f"{rec['chain']},{site_no},{target}\n", encoding="utf-8")
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
        
        text = restore_numbering(produced.read_text(encoding="utf-8", errors="replace"), back)
        problem = check_site(src_lines, text, rec["chain"], rec["resid_struct"], AA1[target])
        if problem:
            return {"mutant": rec["mutant"], "status": "wrong_site", "stderr": problem[:300]}
        produced.write_text(text, encoding="utf-8")
        
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
    parser.add_argument("--force", action="store_true", help="rebuild even when the mutant PDB already exists")
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
    log = [build_one(rec, scap, profix, work, args.force) for rec in rows.to_dict("records")]
    out = Path(args.rows).with_suffix(".scap_log.csv")
    pd.DataFrame(log).to_csv(out, index=False)
    counts = pd.Series([item["status"] for item in log]).value_counts().to_dict()
    print(counts)
    for item in log:
        if item["status"] == "cached":
            print(f"cached {item['stderr']}: {item['mutant']}")
    failed = [item for item in log if item["status"] not in ("ok", "cached")]
    if failed:
        print(failed[0]["mutant"])
        print(failed[0].get("stderr", ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
