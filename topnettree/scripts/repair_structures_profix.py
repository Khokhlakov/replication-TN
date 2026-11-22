#!/usr/bin/env python3
"""Complete missing atoms with Profix BEFORE SCAP, for structures SCAP crashes on.

SCAP segfaults (scap_exit_-11) on some structures right after printing "not complete standard
residue" warnings. 42_build_scap_mutants.py can run Profix, but only on SCAP's output. Here Profix
runs on the SCAP input instead, with the same call 42 uses (`./profix -fix 0 complex.pdb`, in a work
directory that holds its own jackal.dir).

For each structure in the rows file (normally the trimmed, pruned files from make_trimmed_retry.py):
  1. run Profix in a scratch directory;
  2. check the result: no residue added or lost, no chain changed, no input heavy atom lost, and
     report how many incomplete residues remain (a residue that is still incomplete is listed);
  3. write the repaired ATOM records to <out-dir>/<name>.pdb.
Then it writes a rows file whose `structure` is the repaired file (feed it to run_scap_sharded.py) and,
with --manifest, a manifest whose wild_pdb is the repaired file, so wild and mutant are cut from the
same atoms.

    python scripts/repair_structures_profix.py \
        --rows data/features/rows/panel_retry_trimmed.csv \
        --failed-log data/features/rows/panel_retry_trimmed.sharded_scap_log.csv \
        --profix $CAP/data/software/jackal_64bit/bin/profix \
        --manifest data/features/panel_manifest_pruned.csv \
        --out-rows data/features/rows/panel_retry_profix.csv \
        --out-manifest data/features/panel_manifest_profix.csv
"""

from __future__ import annotations

import argparse
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from make_trimmed_retry import find_incomplete, is_h, read_atoms  # noqa: E402


def key(path):
    p = str(path)
    return p[p.index("data/"):] if "data/" in p else p


def run_profix(profix: Path, source: Path, scratch: Path):
    """Same call as 42: work dir with its own jackal.dir, binary copied in, `./profix -fix 0 complex.pdb`."""
    library = profix.resolve().parent.parent / "library"
    if not (library / "tres").is_file():
        raise RuntimeError(f"{library} has no tres; profix is not inside jackal_64bit/bin")
    (scratch / "jackal.dir").write_text(f"pdb: .\nlibrary: {library}\nkey:XIANG2002\n", encoding="utf-8")
    shutil.copyfile(source, scratch / "complex.pdb")
    local = scratch / "profix"
    shutil.copyfile(profix, local)
    local.chmod(0o755)
    proc = subprocess.run(["./profix", "-fix", "0", "complex.pdb"], cwd=scratch, capture_output=True)
    fixed = scratch / "complex_fix.pdb"
    if proc.returncode != 0 or not fixed.is_file():
        tail = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace")[-400:]
        raise RuntimeError(f"profix exit {proc.returncode}: {tail}")
    return fixed


ALL_CHAINS = set(string.ascii_letters + string.digits)


def audit(before_path: Path, after_path: Path):
    b = read_atoms(before_path, ALL_CHAINS, "all")
    a = read_atoms(after_path, ALL_CHAINS, "all")
    res_b = {x["res"] for x in b}
    res_a = {x["res"] for x in a}
    heavy_b = {(x["res"], x["name"]) for x in b if not is_h(x["name"])}
    heavy_a = {(x["res"], x["name"]) for x in a if not is_h(x["name"])}
    return {
        "residues_before": len(res_b),
        "residues_after": len(res_a),
        "residues_lost": sorted(res_b - res_a),
        "residues_new": sorted(res_a - res_b),
        "heavy_lost": len(heavy_b - heavy_a),
        "heavy_added": len(heavy_a - heavy_b),
        "incomplete_before": len(find_incomplete(b)),
        "incomplete_after": find_incomplete(a),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="rows file with a `structure` and a `mutant` column")
    ap.add_argument("--failed-log", help="sharded SCAP log; only rows whose status contains 'scap_exit' are kept")
    ap.add_argument("--profix", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-manifest")
    ap.add_argument("--out-dir", default="data/structures/antibody_rbd_repaired")
    ap.add_argument("--allow-residue-change", action="store_true")
    args = ap.parse_args()

    profix = Path(args.profix)
    if not profix.is_file():
        raise SystemExit(f"profix binary not found: {profix}")
    rows = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
    if args.failed_log:
        log = pd.read_csv(args.failed_log, dtype=str, keep_default_na=False)
        bad = {key(m) for m in log[log["status"].str.contains("scap_exit")]["mutant"]}
        rows = rows[rows["mutant"].map(key).isin(bad)]
    print(f"{len(rows)} rows to repair, {rows['structure'].nunique()} structures")
    if rows.empty:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repaired, report = {}, []
    for structure in rows["structure"].drop_duplicates():
        src = Path(structure) if Path(structure).is_absolute() else ROOT / structure
        dest = out_dir / f"{src.stem}.pdb"
        try:
            with tempfile.TemporaryDirectory(prefix="profix_") as tmp:
                fixed = run_profix(profix, src, Path(tmp))
                info = audit(src, fixed)
                problems = []
                if info["residues_lost"] or info["residues_new"]:
                    problems.append(f"residues changed (lost {len(info['residues_lost'])}, new {len(info['residues_new'])})")
                if info["heavy_lost"]:
                    problems.append(f"{info['heavy_lost']} input heavy atoms lost")
                if problems and not args.allow_residue_change:
                    raise RuntimeError("; ".join(problems))
                lines = [ln for ln in fixed.read_text(encoding="utf-8", errors="replace").splitlines() if ln.startswith("ATOM")]
                dest.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
            repaired[structure] = str(dest)
            left = info["incomplete_after"]
            print(f"  {src.name}: {info['incomplete_before']} incomplete residues before, {len(left)} after; "
                  f"heavy atoms added {info['heavy_added']}")
            for (chain, num, ic), (resname, missing) in list(left.items())[:8]:
                print(f"      still incomplete: {chain}:{resname}{num}{ic} missing {' '.join(missing)}")
            report.append({"structure": structure, "repaired": str(dest), "status": "ok",
                           "incomplete_before": info["incomplete_before"], "incomplete_after": len(left),
                           "heavy_added": info["heavy_added"]})
        except Exception as exc:
            print(f"  {src.name}: FAILED {type(exc).__name__}: {exc}")
            report.append({"structure": structure, "repaired": "", "status": f"failed: {exc}"[:200]})
    pd.DataFrame(report).to_csv(Path(args.out_rows).with_suffix(".repair_report.csv"), index=False, lineterminator="\n")

    ok_rows = rows[rows["structure"].isin(repaired)].copy()
    ok_rows["structure"] = ok_rows["structure"].map(repaired)
    ok_rows.to_csv(args.out_rows, index=False, lineterminator="\n")
    print(f"wrote {args.out_rows}: {len(ok_rows)} rows ({len(rows) - len(ok_rows)} skipped because Profix failed)")

    if args.manifest and args.out_manifest:
        man = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
        lookup = {key(m): s for m, s in zip(ok_rows["mutant"], ok_rows["structure"])}
        man["wild_pdb"] = [lookup.get(key(m), w) for m, w in zip(man["mutant_pdb"], man["wild_pdb"])]
        man.to_csv(args.out_manifest, index=False, lineterminator="\n")
        print(f"wrote {args.out_manifest}: wild_pdb replaced for {len(lookup)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
