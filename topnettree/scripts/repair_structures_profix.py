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
Profix renumbers antibody chains that carry insertion codes (Kabat 82A, 82B, 82C -> 83, 84, 85 and all
residues after them). The residue NAMES stay in the same order, so by default the original chain, number
and insertion code are written back by position (--no-restore-numbering turns that off). Restoration is
refused, and the structure rejected, unless every chain has the same number of residues with the same
names in the same order. The audit then compares atoms residue by residue under the original numbers.
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


def restore_numbering(before_path: Path, fixed_path: Path, dest_path: Path):
    """Write Profix's ATOM records with the input's chain, residue number and insertion code, matched by position.

    Returns (True, message) or (False, reason). Nothing is written unless every chain has the same
    residue names in the same order as the input.
    """
    b = read_atoms(before_path, ALL_CHAINS, "all")
    a_lines = [ln for ln in fixed_path.read_text(encoding="utf-8", errors="replace").splitlines()
               if ln.startswith("ATOM") and len(ln) > 26]

    def order(items):
        seq, seen = {}, set()
        for res, resname in items:
            if res in seen:
                continue
            seen.add(res)
            seq.setdefault(res[0], []).append((res, resname))
        return seq

    in_order = order((x["res"], x["resname"]) for x in b)
    out_items = [((ln[21], ln[22:26].strip(), ln[26].strip()), ln[17:20].strip()) for ln in a_lines]
    out_order = order(out_items)
    mapping = {}
    for chain in sorted(set(in_order) | set(out_order)):
        ri, ro = in_order.get(chain, []), out_order.get(chain, [])
        if [n for _, n in ri] != [n for _, n in ro]:
            return False, f"chain {chain}: {len(ri)} residues in, {len(ro)} out, or the names differ in order; cannot restore numbering"
        for (k_in, _), (k_out, _) in zip(ri, ro):
            mapping[k_out] = k_in
    rewritten = []
    changed = 0
    for ln in a_lines:
        k_out = (ln[21], ln[22:26].strip(), ln[26].strip())
        c, num, ic = mapping[k_out]
        if (c, num, ic) != k_out:
            changed += 1
        rewritten.append(ln[:21] + c + f"{num:>4}" + (ic or " ") + ln[27:])
    dest_path.write_text("\n".join(rewritten) + "\nEND\n", encoding="utf-8")
    return True, f"{len(set(k for k in mapping if mapping[k] != k))} residues renumbered back to the input numbers"


def describe_changes(before_path: Path, after_path: Path, limit: int = 8):
    """Human-readable account of how a Profix output differs from its input."""
    b = read_atoms(before_path, ALL_CHAINS, "all")
    a = read_atoms(after_path, ALL_CHAINS, "all")

    def order(atoms):
        seq, seen = {}, set()
        for x in atoms:
            if x["res"] in seen:
                continue
            seen.add(x["res"])
            seq.setdefault(x["chain"], []).append((x["res"], x["resname"]))
        return seq

    ob, oa = order(b), order(a)
    out = []
    for chain in sorted(set(ob) | set(oa)):
        rb, ra = ob.get(chain, []), oa.get(chain, [])
        names_equal = [n for _, n in rb] == [n for _, n in ra]
        moved = [(k1, k2) for (k1, n1), (k2, n2) in zip(rb, ra) if k1 != k2]
        out.append(f"chain {chain}: {len(rb)} residues in, {len(ra)} out, residue names in the same order: {names_equal}, "
                   f"{len(moved)} with a different number")
        if moved:
            shown = ", ".join(f"{k1[1]}{k1[2]}->{k2[1]}{k2[2]}" for k1, k2 in moved[:limit])
            out.append(f"    renumbered, first {min(limit, len(moved))}: {shown}")
    by_res_b = {}
    for x in b:
        if not is_h(x["name"]):
            by_res_b.setdefault(x["res"], set()).add(x["name"])
    by_res_a = {}
    for x in a:
        if not is_h(x["name"]):
            by_res_a.setdefault(x["res"], set()).add(x["name"])
    renamed = [(k, sorted(v - by_res_a[k])[:4], sorted(by_res_a[k] - v)[:4]) for k, v in by_res_b.items()
               if k in by_res_a and (v - by_res_a[k])]
    out.append(f"{len(renamed)} residues keep their number but lose or rename heavy atoms")
    for k, lost, new in renamed[:limit]:
        out.append(f"    {k[0]}:{k[1]}{k[2]} lost {' '.join(lost)}; output has {' '.join(new) or '-'}")
    return out


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
    ap.add_argument("--no-restore-numbering", action="store_true")
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
                raw_copy = out_dir / f"{src.stem}.profix_raw.pdb"
                shutil.copyfile(fixed, raw_copy)
                if not args.no_restore_numbering:
                    restored = Path(tmp) / "restored.pdb"
                    ok, msg = restore_numbering(src, fixed, restored)
                    print(f"  {src.name}: {msg}")
                    if ok:
                        fixed = restored
                info = audit(src, fixed)
                problems = []
                if info["residues_lost"] or info["residues_new"]:
                    problems.append(f"residues changed (lost {len(info['residues_lost'])}, new {len(info['residues_new'])})")
                if info["heavy_lost"]:
                    problems.append(f"{info['heavy_lost']} input heavy atoms lost")
                if problems and not args.allow_residue_change:
                    print(f"  {src.name}: REJECTED, {'; '.join(problems)}. Raw Profix output kept at {raw_copy}")
                    for line in describe_changes(src, fixed):
                        print(f"      {line}")
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
