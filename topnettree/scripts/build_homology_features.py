#!/usr/bin/env python3
"""Batch H0 and H1/H2 features for a table of mutations.

Three steps. Run them in this order; each one is useful on its own.

  check     Look at the data only. No homology is computed. Finds missing files,
            wrong residue letters, mutants that differ from the wild type
            outside the mutated residue, empty neighborhoods, and unexpected
            partner chains. Writes check_report.csv. Exit status 1 if any row
            fails.
  build     Compute the blocks row by row (resumable, parallel). A row that
            fails is recorded and left as NaN. It is never filled with zeros.
  validate  Check the written arrays against the invariants that the layout
            implies, and optionally compare them with a reference tensor
            (for example the shipped S645 X_unnormed.npy).

Manifest CSV (one row per mutation). Required columns:

    row_id      file-safe key (letters, digits, _ . -)
    wild_pdb    path to the wild-type complex
    mutant_pdb  path to the mutant complex
    chain       chain id of the mutated residue (one character)
    resid       residue number, insertion code allowed (100A)

Optional columns: icode, wt (one letter), mut (one letter), keep_chains
("H+L+E", the chains of the complex; everything else in the file is dropped
before the neighborhoods are cut). Use --map name=column when your table calls
a column something else, e.g. --map resid=resid_str --map wild_pdb=file.

    python scripts/build_homology_features.py check    --manifest m.csv --out-dir features/panel
    python scripts/build_homology_features.py build    --manifest m.csv --out-dir features/panel --jobs 8
    python scripts/build_homology_features.py validate --out-dir features/panel [--reference X.npy]

Output directory:
    rows/<row_id>_h0.npy, rows/<row_id>_h12.npy   one file per finished row
    h0.npy       (N, 48, 108) float64, NaN rows where ok_mask is False
    h12.npy      (N, 756) float64, NaN rows where ok_mask is False (not with --h0-only)
    ok_mask.npy  (N,) bool
    index.csv    row_id, status, error, seconds
    run_summary.json   options, versions, manifest hash, generator hash
"""

from __future__ import annotations

import os

# One BLAS thread per worker. Set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import re
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from topnet.features import homology as hom
from topnet.features.manifest import GENERATOR_MANIFEST, manifest_hash
from topnet.features.pdbio import AA1, read_pdb_records

H0_SHAPE = (48, 108)
H12_DIM = 756
SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")
REQUIRED = ("row_id", "wild_pdb", "mutant_pdb", "chain", "resid")
OPTIONAL = ("icode", "wt", "mut", "keep_chains")
BIG_NEIGHBORHOOD = 8000  # C/N/O atoms in one neighborhood; slow beyond this


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def load_manifest(path, mapping):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    rename = {col: name for name, col in mapping.items()}
    frame = frame.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise SystemExit(f"manifest {path} lacks columns {missing}; has {list(frame.columns)}. Use --map.")
    for col in OPTIONAL:
        if col not in frame.columns:
            frame[col] = ""
    frame = frame.reset_index(drop=True)
    bad = [r for r in frame["row_id"] if not SAFE_ID.match(r)]
    if bad:
        raise SystemExit(f"row_id values that are not file-safe: {bad[:5]}")
    dup = frame["row_id"][frame["row_id"].duplicated()].tolist()
    if dup:
        raise SystemExit(f"duplicate row_id values: {dup[:5]}")
    return frame


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# structure cache (per process)
# --------------------------------------------------------------------------
class RecordCache:
    def __init__(self, size=4):
        self.size = size
        self.items = OrderedDict()

    def get(self, path, altloc):
        key = (str(path), altloc)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"file not found: {path}")
        records = read_pdb_records(path, altloc=altloc)
        if not records:
            raise ValueError(f"no ATOM records in {path}")
        self.items[key] = records
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        return records


_CACHE = RecordCache()


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def _heavy_count(records):
    return len(hom._heavy(records))


def audit_row(row, opts):
    """Return (status, messages, stats) for one manifest row. No homology."""
    fails, warns, stats = [], [], {}
    try:
        number, icode = hom.parse_site(row["resid"], row["icode"] or None)
        chain = row["chain"]
        if len(chain) != 1:
            return "fail", [f"chain {chain!r} is not one character"], stats
        keep = hom.parse_keep_chains(row["keep_chains"])
        wild = _CACHE.get(row["wild_pdb"], opts["altloc"])
        mutant = _CACHE.get(row["mutant_pdb"], opts["altloc"])
        stats["chains_in_file"] = "".join(sorted({r["chain"] for r in wild}))
        wild = hom.restrict_to_chains(wild, keep, chain, "wild")
        mutant = hom.restrict_to_chains(mutant, keep, chain, "mutant")
    except ValueError as exc:
        return "fail", [str(exc)], stats

    legacy = opts["legacy_resid"]
    site = {}
    for label, recs in (("wild", wild), ("mutant", mutant)):
        at_site = [r for r in recs if hom._is_site(r, chain, number, icode, legacy)]
        names = sorted({r["resname"] for r in at_site})
        site[label] = (at_site, names)
        stats[f"{label}_site_atoms"] = len(at_site)
        if not at_site:
            fails.append(f"{label}: residue {chain}{number}{icode} not found")
        elif len(names) > 1:
            fails.append(f"{label}: residue key {chain}{number}{icode} matches several residue names {names}")
    if not fails:
        wt_name = site["wild"][1][0]
        mut_name = site["mutant"][1][0]
        stats["wild_resname"], stats["mutant_resname"] = wt_name, mut_name
        if row["wt"] and AA1.get(row["wt"].upper()) != wt_name:
            fails.append(f"wild residue is {wt_name}, table says {row['wt']}")
        if row["mut"] and AA1.get(row["mut"].upper()) != mut_name:
            fails.append(f"mutant residue is {mut_name}, table says {row['mut']}")
        if wt_name == mut_name:
            warns.append("wild and mutant residue names are identical")

        # Everything outside the site should be the same atoms in the same order.
        w_rest = [r for r in wild if not hom._is_site(r, chain, number, icode, legacy)]
        m_rest = [r for r in mutant if not hom._is_site(r, chain, number, icode, legacy)]
        stats["outside_atoms_wild"], stats["outside_atoms_mutant"] = len(w_rest), len(m_rest)
        if len(w_rest) != len(m_rest):
            warns.append(f"atoms outside the site differ: wild {len(w_rest)}, mutant {len(m_rest)} (hydrogens added or lost?)")
        else:
            same_keys = all(
                a["chain"] == b["chain"] and a["resseq"] == b["resseq"] and a["icode"] == b["icode"] and a["element"] == b["element"]
                for a, b in zip(w_rest, m_rest)
            )
            if not same_keys:
                warns.append("atoms outside the site are in a different order or have different identities")
            else:
                shift = np.linalg.norm(
                    np.asarray([r["xyz"] for r in w_rest]) - np.asarray([r["xyz"] for r in m_rest]), axis=1
                )
                stats["max_shift_outside"] = float(shift.max()) if len(shift) else 0.0
                moved = int((shift > 0.05).sum())
                if moved:
                    warns.append(f"{moved} atoms outside the site moved more than 0.05 A (max {shift.max():.2f})")

        # Neighborhoods and the chains that end up as partner.
        for label, recs in (("wild", wild), ("mutant", mutant)):
            bind = hom.binding_neighborhood(recs, chain, opts["cutoff"])
            b_a, b_b = hom.split_binding(bind, chain)
            mutn = hom.mutation_neighborhood(recs, chain, number, icode, opts["cutoff"], legacy)
            m_a, m_b = hom.split_mutation(mutn, chain, number, icode, legacy)
            counts = [_heavy_count(x) for x in (b_a, b_b, m_a, m_b)]
            stats[f"{label}_bind_heavy"] = f"{counts[0]}/{counts[1]}"
            stats[f"{label}_site_heavy"] = f"{counts[2]}/{counts[3]}"
            if min(counts) == 0:
                fails.append(f"{label}: a neighborhood side has no C/N/O atoms (bind {counts[:2]}, site {counts[2:]})")
            if max(counts) > BIG_NEIGHBORHOOD:
                warns.append(f"{label}: large neighborhood ({max(counts)} C/N/O atoms), slow")
            if label == "wild":
                stats["partner_chains"] = "".join(sorted({r["chain"] for r in b_b}))
    status = "fail" if fails else ("warn" if warns else "ok")
    return status, fails + warns, stats


def _audit_worker(args):
    row, opts = args
    try:
        return audit_row(row, opts)
    except Exception as exc:  # report, never crash the whole run
        return "fail", [f"{type(exc).__name__}: {exc}"], {}


def cmd_check(args):
    frame = load_manifest(args.manifest, parse_map(args.map))
    if args.limit:
        frame = frame.head(args.limit)
    opts = options_from(args)
    order = frame.sort_values(["wild_pdb", "mutant_pdb"]).index
    jobs = [(frame.loc[i].to_dict(), opts) for i in order]
    results = run_pool(_audit_worker, jobs, args.jobs)
    rows = []
    for (row, _), (status, msgs, stats) in zip(jobs, results):
        rows.append({"row_id": row["row_id"], "status": status, "messages": " | ".join(msgs), **stats})
    report = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "check_report.csv", index=False)
    tally = report["status"].value_counts().to_dict()
    print(f"check: {tally}  ->  {out / 'check_report.csv'}")
    for status in ("fail", "warn"):
        sub = report[report["status"] == status]
        if len(sub):
            print(f"\n{status.upper()} ({len(sub)} rows, first 15):")
            for _, r in sub.head(15).iterrows():
                print(f"  {r['row_id']}: {r['messages']}")
    return 1 if tally.get("fail", 0) else 0


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def _build_worker(args):
    row, opts, rows_dir = args
    rid = row["row_id"]
    h0_path = Path(rows_dir) / f"{rid}_h0.npy"
    h12_path = Path(rows_dir) / f"{rid}_h12.npy"
    done = h0_path.is_file() and (opts["h0_only"] or h12_path.is_file())
    if done and not opts["force"]:
        return rid, "ok", "", 0.0
    t0 = time.time()
    try:
        wild = _CACHE.get(row["wild_pdb"], opts["altloc"])
        mutant = _CACHE.get(row["mutant_pdb"], opts["altloc"])
        blocks = hom.homology_blocks(
            wild,
            mutant,
            row["chain"],
            row["resid"],
            row["icode"] or None,
            cutoff=opts["cutoff"],
            legacy_resid=opts["legacy_resid"],
            quantize=not opts["no_quantize"],
            compute_h12=not opts["h0_only"],
            altloc=opts["altloc"],
            keep_chains=row["keep_chains"] or None,
        )
        np.save(h0_path, blocks.h0)
        if blocks.h12 is not None:
            np.save(h12_path, blocks.h12)
        return rid, "ok", "", time.time() - t0
    except ImportError as exc:
        return rid, "fail", f"ImportError: {exc}", time.time() - t0
    except Exception as exc:
        return rid, "fail", f"{type(exc).__name__}: {exc}", time.time() - t0


def cmd_build(args):
    if not args.h0_only:
        try:
            import gudhi  # noqa: F401
        except ImportError:
            raise SystemExit("gudhi is not installed (pip install gudhi). Use --h0-only to skip H1/H2.")
    frame = load_manifest(args.manifest, parse_map(args.map))
    if args.limit:
        frame = frame.head(args.limit)
    opts = options_from(args)
    out = Path(args.out_dir)
    rows_dir = out / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    # Group rows of one wild-type file together so the per-process cache hits.
    order = frame.sort_values(["wild_pdb", "mutant_pdb"]).index
    jobs = [(frame.loc[i].to_dict(), opts, str(rows_dir)) for i in order]
    t0 = time.time()
    results = run_pool(_build_worker, jobs, args.jobs, progress=True)
    status = {rid: (st, err, sec) for rid, st, err, sec in results}

    n = len(frame)
    h0 = np.full((n,) + H0_SHAPE, np.nan, dtype=np.float64)
    h12 = None if args.h0_only else np.full((n, H12_DIM), np.nan, dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    index = []
    for i, rid in enumerate(frame["row_id"]):
        st, err, sec = status[rid]
        if st == "ok":
            a = np.load(rows_dir / f"{rid}_h0.npy")
            if a.shape != H0_SHAPE:
                st, err = "fail", f"h0 shape {a.shape}"
            elif h12 is not None:
                b = np.load(rows_dir / f"{rid}_h12.npy")
                if b.shape != (H12_DIM,):
                    st, err = "fail", f"h12 shape {b.shape}"
                else:
                    h12[i] = b
            if st == "ok":
                h0[i] = a
                ok[i] = True
        index.append({"row_idx": i, "row_id": rid, "status": st, "error": err, "seconds": round(sec, 3)})
    np.save(out / "h0.npy", h0)
    if h12 is not None:
        np.save(out / "h12.npy", h12)
    np.save(out / "ok_mask.npy", ok)
    pd.DataFrame(index).to_csv(out / "index.csv", index=False)

    versions = {"numpy": np.__version__}
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        import gudhi
        versions["gudhi"] = gudhi.__version__
    except ImportError:
        versions["gudhi"] = None
    summary = {
        "n_rows": n,
        "n_ok": int(ok.sum()),
        "n_failed": int((~ok).sum()),
        "options": opts,
        "versions": versions,
        "manifest_csv": str(args.manifest),
        "manifest_csv_sha256": file_sha256(args.manifest),
        "generator_manifest_id": GENERATOR_MANIFEST["id"],
        "generator_manifest_hash": manifest_hash(GENERATOR_MANIFEST),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"build: {int(ok.sum())}/{n} ok, {int((~ok).sum())} failed  ->  {out}")
    failed = [r for r in index if r["status"] != "ok"]
    for r in failed[:15]:
        print(f"  FAILED {r['row_id']}: {r['error']}")
    return 1 if failed else 0


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
GROUPS = ("bind_death", "bind_persist", "mut_death", "mut_persist")
PARTS = ("wild", "mutant", "wild_minus_mutant")


def h0_invariants(h0):
    """Violation counts for the (N, 48, 108) tensor. Rows must be finite."""
    n = len(h0)
    out = {}
    blocks = h0.reshape(n, 48, 4, 3, 9)  # bin, group, part, pair
    wild, mut, diff = blocks[:, :, :, 0], blocks[:, :, :, 1], blocks[:, :, :, 2]
    out["diff_is_wild_minus_mutant"] = int((np.abs(diff - (wild - mut)) > 1e-9).any(axis=(1, 2, 3)).sum())
    out["negative_counts"] = int(((wild < 0) | (mut < 0)).any(axis=(1, 2, 3)).sum())
    out["non_integer_counts"] = int(((np.abs(wild - np.round(wild)) > 1e-9) | (np.abs(mut - np.round(mut)) > 1e-9)).any(axis=(1, 2, 3)).sum())
    bad = 0
    for part in (wild, mut):
        for g in (0, 2):  # death group; persistence group is g + 1
            death, persist = part[:, :, g], part[:, :, g + 1]
            # persistence[j] = number of bars that die in a later bin = sum of death[b > j]
            later = np.cumsum(death[:, ::-1], axis=1)[:, ::-1] - death
            bad += int((np.abs(persist - later) > 1e-9).any(axis=(1, 2)).sum())
    out["persistence_not_tail_sum_of_deaths"] = bad
    return out


def h12_invariants(h12):
    n = len(h12)
    out = {}
    blocks = h12.reshape(n, 2, 2, 3, 3, 7, 3)  # site, dim, row, col, stat, (wild, mutant, diff)
    wild, mut, diff = blocks[..., 0], blocks[..., 1], blocks[..., 2]
    out["diff_is_mutant_minus_wild"] = int((np.abs(diff - (mut - wild)) > 1e-9).any(axis=(1, 2, 3, 4, 5)).sum())
    bad = 0
    for part in (wild, mut):
        s, mx, mean = part[..., 0], part[..., 1], part[..., 2]
        nonzero = s != 0
        bad += int((nonzero & ((s < mx - 1e-9) | (mx < mean - 1e-9) | (mean < -1e-9))).any(axis=(1, 2, 3, 4)).sum())
        bad += int((nonzero & ((part[..., 3] > part[..., 4] + 1e-9) | (part[..., 5] > part[..., 6] + 1e-9))).any(axis=(1, 2, 3, 4)).sum())
    out["bar_stat_order_violations"] = bad
    return out


def cmd_validate(args):
    out = Path(args.out_dir)
    ok = np.load(out / "ok_mask.npy")
    h0 = np.load(out / "h0.npy")
    print(f"rows {len(ok)}, ok {int(ok.sum())}")
    problems = 0
    if h0.shape[1:] != H0_SHAPE:
        print(f"FAIL h0 shape {h0.shape}")
        return 1
    good = h0[ok]
    if not np.isfinite(good).all():
        print("FAIL non-finite values in rows marked ok")
        problems += 1
    if (~ok).any() and np.isfinite(h0[~ok]).any():
        print("FAIL failed rows are not NaN")
        problems += 1
    inv = h0_invariants(good)
    print("H0 invariants (rows violating):", inv)
    problems += sum(inv.values())
    parts = good.reshape(len(good), 48, 4, 3, 9)[:, :, :, :2]  # wild and mutant, not the difference
    empty = (parts == 0).all(axis=1)  # (rows, group, part, pair): a column with no bar at all
    print(f"H0 wild/mutant columns with no bars, per row: mean {empty.sum(axis=(1, 2, 3)).mean():.2f}, max {int(empty.sum(axis=(1, 2, 3)).max())}"
          " (nonzero means an element pair had no atoms on one side)")
    deaths = good.reshape(len(good), 48, 4, 3, 9)[:, :, 0, 0]  # binding death, wild
    print("binding death count per pair, wild, mean over rows:", np.round(deaths.sum(axis=1).mean(axis=0), 1).tolist())

    h12_path = out / "h12.npy"
    if h12_path.is_file():
        h12 = np.load(h12_path)[ok]
        if h12.shape[1:] != (H12_DIM,) or not np.isfinite(h12).all():
            print(f"FAIL h12 shape {h12.shape} or non-finite")
            problems += 1
        else:
            inv12 = h12_invariants(h12)
            print("H12 invariants (rows violating):", inv12)
            problems += sum(inv12.values())
            print(f"H12 rows that are entirely zero: {int((h12 == 0).all(axis=1).sum())}")

    if args.reference:
        ref = np.load(args.reference)
        if ref.ndim != 3 or ref.shape[1:] != H0_SHAPE:
            print(f"reference shape {ref.shape} is not (N, 48, 108)")
            return 1
        idx = pd.read_csv(out / "index.csv")
        if args.ref_rows:
            ref_rows = np.loadtxt(args.ref_rows, dtype=int)
        else:
            ref_rows = idx["row_idx"].to_numpy()
        if len(ref_rows) != len(ok):
            print("reference row list length does not match the manifest")
            return 1
        sel = ok & (ref_rows >= 0) & (ref_rows < len(ref))
        mine = h0[sel].reshape(-1, 48, 4, 3, 9)
        theirs = ref[ref_rows[sel]].reshape(-1, 48, 4, 3, 9)
        print(f"\ncompared with {args.reference} on {int(sel.sum())} rows")
        print(f"{'group':14s}{'part':20s}{'rows equal':>12s}{'mean|diff|':>12s}{'max|diff|':>11s}")
        for g, gname in enumerate(GROUPS):
            for p, pname in enumerate(PARTS):
                d = np.abs(mine[:, :, g, p] - theirs[:, :, g, p])
                equal = float((d.max(axis=(1, 2)) < 1e-9).mean())
                print(f"{gname:14s}{pname:20s}{equal:12.3f}{d.mean():12.4f}{d.max():11.1f}")
    print("\nvalidate:", "OK" if problems == 0 else f"{problems} problem(s)")
    return 1 if problems else 0


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------
def parse_map(items):
    mapping = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--map wants name=column, got {item!r}")
        name, col = item.split("=", 1)
        if name not in REQUIRED + OPTIONAL:
            raise SystemExit(f"--map name must be one of {REQUIRED + OPTIONAL}, got {name!r}")
        mapping[name] = col
    return mapping


def options_from(args):
    return {
        "cutoff": args.cutoff,
        "legacy_resid": args.legacy_resid,
        "no_quantize": args.no_quantize,
        "altloc": args.altloc,
        "h0_only": getattr(args, "h0_only", False),
        "force": getattr(args, "force", False),
    }


def run_pool(func, jobs, n_jobs, progress=False):
    total = len(jobs)
    results = []
    t0 = time.time()
    if n_jobs <= 1:
        iterator = map(func, jobs)
        pool = None
    else:
        pool = Pool(n_jobs)
        iterator = pool.imap(func, jobs, chunksize=4)  # keeps file order, so cache hits stay local
    try:
        for i, res in enumerate(iterator, 1):
            results.append(res)
            if progress and (i % 25 == 0 or i == total):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{total}  {rate:.2f} rows/s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return results


def common(parser, needs_manifest=True):
    if needs_manifest:
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--map", action="append", help="name=column, repeatable")
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--jobs", type=int, default=1)
        parser.add_argument("--cutoff", type=float, default=12.0)
        parser.add_argument("--legacy-resid", action="store_true")
        parser.add_argument("--altloc", choices=["all", "blank", "first"], default="all")
    parser.add_argument("--out-dir", required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    common(p)
    p.set_defaults(func=cmd_check, no_quantize=False)
    p = sub.add_parser("build")
    common(p)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--h0-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("validate")
    common(p, needs_manifest=False)
    p.add_argument("--reference", help="(N,48,108) tensor to compare with, e.g. data/raw/X_unnormed.npy")
    p.add_argument("--ref-rows", help="text file: reference row index for each manifest row (default 0..N-1)")
    p.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
