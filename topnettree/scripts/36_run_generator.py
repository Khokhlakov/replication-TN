#!/usr/bin/env python3
"""Steps 2 and 6. Run an external per-mutation feature generator over a table of rows. Resumable, parallel, logged.

The generator is whatever your capsule script is (aux, H1/H2, H0). It is called once per row through a command template;
this runner does not know its internals. Placeholders: {row} {pdb} {chain} {wt} {resid} {mut} {structure} {out}.
The command must write ONE vector for the row to {out} (.npy, or whitespace/comma text with --out-format txt).

    python scripts/36_run_generator.py \\
        --rows data/raw/skempi/s1131_joined.csv --structure-dir data/raw/skempi/pdbs \\
        --cmd "python CAPSULE/code/gen_feature_aux/run_one.py --pdb {structure} --chain {chain} --wt {wt} --resid {resid} --mut {mut} --out {out}" \\
        --out-dir data/features/s1131_aux --workers 8 --timeout 1800

Structure path: --structure-dir/<PDB>.pdb, or a 'structure' column in --rows (takes precedence).
Outputs in --out-dir: rows/<k>.npy (cache; existing files are reused), features_full.npy (NaN rows for failures, alignment
with --rows preserved), ok_mask.npy, index.csv (status, seconds, stderr tail), run_summary.json (sha256 of features_full).
Failed rows are never imputed. Rerun to retry failures (only successful rows are cached).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def run_row(k: int, rec: dict, args, out_dir: Path) -> dict:
    cache = out_dir / "rows" / f"{k}.npy"
    if cache.is_file() and not args.force:
        return {"row": k, "status": "cached", "seconds": 0.0, "stderr": ""}
    structure = rec.get("structure") or (str(Path(args.structure_dir) / f"{str(rec['pdb']).upper()}.pdb") if args.structure_dir else "")
    if structure and not Path(structure).is_file():
        return {"row": k, "status": "missing_structure", "seconds": 0.0, "stderr": structure}
    tmp = out_dir / "rows" / f"{k}.out"
    fields = {"row": k, "pdb": rec.get("pdb", ""), "chain": rec.get("chain", rec.get("chain_s", "")), "wt": rec.get("wt", ""),
              "resid": rec.get("resid_struct", rec.get("resid_s", rec.get("resid", ""))), "mut": rec.get("mut", ""),
              "structure": structure, "out": str(tmp)}
    argv = [tok.format(**fields) for tok in shlex.split(args.cmd)]
    t0 = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=args.timeout, cwd=args.cwd)
    except subprocess.TimeoutExpired:
        return {"row": k, "status": "timeout", "seconds": time.time() - t0, "stderr": ""}
    dt = time.time() - t0
    alt = Path(str(tmp) + ".npy")  # np.save appends .npy when the name lacks it
    if not tmp.is_file() and alt.is_file():
        tmp = alt
    if proc.returncode != 0 or not tmp.is_file():
        return {"row": k, "status": f"exit_{proc.returncode}", "seconds": dt, "stderr": proc.stderr[-300:].replace("\n", " | ")}
    try:
        vec = np.load(tmp, allow_pickle=False) if args.out_format == "npy" else np.loadtxt(tmp, delimiter="," if "," in tmp.read_text()[:200] else None)
        vec = np.asarray(vec, dtype=np.float64).ravel()
    except Exception as err:
        return {"row": k, "status": "unreadable_output", "seconds": dt, "stderr": str(err)[:200]}
    if not np.isfinite(vec).all():
        return {"row": k, "status": "non_finite", "seconds": dt, "stderr": f"{int((~np.isfinite(vec)).sum())} non-finite values"}
    np.save(cache, vec)
    tmp.unlink(missing_ok=True)
    return {"row": k, "status": "ok", "seconds": dt, "stderr": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--structure-dir")
    ap.add_argument("--cwd")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out-format", choices=["npy", "txt"], default="npy")
    ap.add_argument("--expect-dim", type=int)
    ap.add_argument("--limit", type=int, help="run only the first N rows (smoke test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "rows").mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
    if args.limit:
        rows = rows.iloc[: args.limit]
    recs = rows.to_dict("records")
    log = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_row, k, rec, args, out_dir): k for k, rec in enumerate(recs)}
        for i, fut in enumerate(as_completed(futs), 1):
            log.append(fut.result())
            if i % 100 == 0:
                print(f"  {i}/{len(recs)}")
    index = pd.DataFrame(sorted(log, key=lambda d: d["row"]))

    vecs, ok = [], np.zeros(len(recs), dtype=bool)
    dim = args.expect_dim
    for k in range(len(recs)):
        f = out_dir / "rows" / f"{k}.npy"
        if f.is_file():
            v = np.load(f)
            dim = dim or len(v)
            if len(v) == dim:
                ok[k] = True
                vecs.append(v)
                continue
            index.loc[index.row == k, "status"] = f"bad_dim_{len(v)}"
        vecs.append(None)
    if dim is None:
        raise SystemExit("No row produced a feature vector. See index.csv for the statuses and stderr tails.")
    full = np.full((len(recs), dim), np.nan)
    for k, v in enumerate(vecs):
        if v is not None:
            full[k] = v
    np.save(out_dir / "features_full.npy", full)
    np.save(out_dir / "ok_mask.npy", ok)
    index.to_csv(out_dir / "index.csv", index=False)
    summary = {"n_rows": len(recs), "n_ok": int(ok.sum()), "dim": int(dim),
               "status_counts": index["status"].value_counts().to_dict(),
               "sha256_features_full": hashlib.sha256(full.tobytes()).hexdigest()}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if (~ok).any():
        print("failures (first 10):")
        print(index[index.status.isin(index.status[~index.status.isin(["ok", "cached"])])].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
