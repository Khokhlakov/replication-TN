#!/usr/bin/env python3
"""Run 42_build_scap_mutants.py on several shards of one rows file in parallel.

42 builds one mutant at a time. This script
  1. reads the rows file and drops duplicate `mutant` paths (two processes must never write the
     same file) and mutants that already exist, so a restart only builds what is missing;
  2. splits the rest into --shards disjoint files;
  3. starts 42 once per shard, each with its own --work directory (42 names its scratch
     directory after the mutant and deletes it afterwards, so shards must not share one);
  4. waits, printing progress, and merges the shard logs into <rows>.sharded_scap_log.csv.

42 itself is not modified. Failed rows are listed, never replaced by an invented file.

    python scripts/run_scap_sharded.py --rows data/features/rows/starr2020.csv \
        --scap $CAP/data/software/jackal_64bit/bin/scap --shards 16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def usable(path: Path) -> bool:
    """Same test 42 uses to decide a mutant is already built."""
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    with path.open("rb") as handle:
        return b"ATOM" in handle.read(4000)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--scap", required=True)
    ap.add_argument("--profix")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N missing mutants (for a trial)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--script", default=str(ROOT / "scripts" / "42_build_scap_mutants.py"))
    ap.add_argument("--work-root", default="/tmp/scap_work", help="one subdirectory per shard is made here")
    ap.add_argument("--shard-dir", help="default: <rows folder>/<rows stem>_shards")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    rows_path = Path(args.rows)
    rows = pd.read_csv(rows_path, dtype=str, keep_default_na=False)
    if "mutant" not in rows.columns:
        raise SystemExit(f"{rows_path} has no `mutant` column")
    n_all = len(rows)
    rows = rows.drop_duplicates("mutant")
    n_unique = len(rows)
    if not args.force:
        rows = rows[[not usable(ROOT / m) for m in rows["mutant"]]]
    if args.limit:
        rows = rows.head(args.limit)
    print(f"{n_all} rows, {n_unique} distinct mutants, {len(rows)} to build")
    if rows.empty:
        print("nothing to do")
        return 0

    shards = max(1, min(args.shards, len(rows)))
    shard_dir = Path(args.shard_dir) if args.shard_dir else rows_path.parent / f"{rows_path.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root)
    procs, shard_files, shard_targets = [], [], []
    for k in range(shards):
        part = rows.iloc[k::shards]
        f = shard_dir / f"shard_{k:02d}.csv"
        part.to_csv(f, index=False, lineterminator="\n")
        shard_files.append(f)
        shard_targets.append([ROOT / m for m in part["mutant"]])
    t0 = time.time()
    try:
        for k, f in enumerate(shard_files):
            cmd = [sys.executable, args.script, "--rows", str(f), "--scap", args.scap,
                   "--work", str(work_root / f"shard_{k:02d}")]
            if args.profix:
                cmd += ["--profix", args.profix]
            if args.force:
                cmd.append("--force")
            log = open(shard_dir / f"shard_{k:02d}.out", "w")
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        total = len(rows)
        while any(p.poll() is None for p, _ in procs):
            time.sleep(args.poll)
            done = sum(usable(t) for targets in shard_targets for t in targets)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (total - done) / rate / 60 if rate > 0 else float("nan")
            print(f"  {done}/{total} built, {rate:.2f}/s, about {eta:.0f} min left", flush=True)
    except KeyboardInterrupt:
        for p, _ in procs:
            p.terminate()
        print("interrupted; finished mutants are kept and a restart skips them")
        return 130
    finally:
        for _, log in procs:
            log.close()

    frames = []
    for f in shard_files:
        log_csv = f.with_suffix(".scap_log.csv")
        if log_csv.is_file():
            frames.append(pd.read_csv(log_csv, dtype=str, keep_default_na=False))
    codes = [p.returncode for p, _ in procs]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["mutant", "status", "stderr"])
    out = rows_path.with_suffix(".sharded_scap_log.csv")
    merged.to_csv(out, index=False, lineterminator="\n")
    counts = merged["status"].value_counts().to_dict() if len(merged) else {}
    print(f"finished in {(time.time() - t0) / 60:.1f} min: {counts}; shard exit codes {sorted(set(codes))}")
    bad = merged[~merged["status"].isin(["ok", "cached"])] if len(merged) else merged
    for _, r in bad.head(10).iterrows():
        print(f"  FAILED {r['mutant']}: {r['status']} {r['stderr'][:200]}")
    print(f"wrote {out}")
    return 1 if (len(bad) or any(codes)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
