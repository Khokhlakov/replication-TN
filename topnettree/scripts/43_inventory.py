#!/usr/bin/env python
"""What is ready for the H0 / H1 / H2 stage?  Read-only audit + text bundles.

Run from the repo root (tcsh):

    python scripts/43_inventory.py --capsule /path/to/capsule >& logs/inventory.log
    python scripts/43_inventory.py --capsule /path/to/capsule --bundle >& logs/bundle.log

Writes only logs/inventory.json and (with --bundle) logs/bundle_repo.txt and
logs/bundle_capsule.txt.  Every check is wrapped, so one missing file never
stops the rest of the report.  Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = {}


# ----------------------------------------------------------------- helpers
def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def guarded(name):
    """Decorator: run a check, print and record any exception instead of dying."""
    def deco(fn):
        def run(*a, **k):
            section(name)
            try:
                out = fn(*a, **k)
                REPORT[name] = out
                return out
            except Exception as exc:  # noqa: BLE001
                msg = "%s: %s" % (type(exc).__name__, exc)
                print("  !! check failed -> " + msg)
                REPORT[name] = {"error": msg}
        return run
    return deco


def pick(cols, *cands):
    low = {str(c).lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def csv_info(path, show_head=2):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    info = {"path": str(path.relative_to(ROOT)), "rows": int(len(df)), "columns": list(df.columns)}
    print("  %-52s rows=%-6d cols=%s" % (info["path"], len(df), list(df.columns)))
    for _, r in df.head(show_head).iterrows():
        print("      " + " | ".join("%s=%s" % (k, str(v)[:28]) for k, v in r.items()))
    return info, df


def npy_info(path):
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"path": str(path), "shape": list(arr.shape), "dtype": str(arr.dtype)}


def count(pattern):
    return len(list(ROOT.glob(pattern)))


# ----------------------------------------------------------------- checks
@guarded("1. labels")
def check_labels():
    out = {}
    for p in sorted((ROOT / "data/labels").glob("*.csv")):
        try:
            info, df = csv_info(p)
        except Exception as exc:  # noqa: BLE001
            out[p.name] = {"error": str(exc)}
            print("  !! %s: %s" % (p.name, exc))
            continue
        out[p.name] = info
        for col in ("source", "dataset", "set", "origin", "reason"):
            c = pick(df.columns, col)
            if c is not None:
                vc = df[c].value_counts().to_dict()
                print("      %s counts: %s" % (c, vc))
                info["counts_" + c] = {str(k): int(v) for k, v in vc.items()}
    return out


@guarded("2. feature row tables (what SHOULD be featurised)")
def check_rows():
    out = {}
    muts = {p.name for p in (ROOT / "data/features/mutants").glob("*.pdb")}
    for p in sorted((ROOT / "data/features/rows").glob("*.csv")):
        try:
            info, df = csv_info(p)
        except Exception as exc:  # noqa: BLE001
            out[p.name] = {"error": str(exc)}
            print("  !! %s: %s" % (p.name, exc))
            continue
        c_pdb, c_ch = pick(df.columns, "pdb", "pdb_id"), pick(df.columns, "chain")
        c_wt, c_res, c_mut = pick(df.columns, "wt"), pick(df.columns, "resid", "resid_str"), pick(df.columns, "mut")
        c_file = pick(df.columns, "file", "pdb_file", "structure", "path")
        if c_file is not None:
            missing = [f for f in df[c_file] if f and not (ROOT / f).is_file() and not Path(f).is_file()]
            info["structure_files_missing"] = len(missing)
            print("      structure file column '%s': %d of %d paths not found" % (c_file, len(missing), len(df)))
        if all(c is not None for c in (c_pdb, c_ch, c_wt, c_res, c_mut)):
            names = (df[c_pdb] + "_" + df[c_ch] + "_" + df[c_wt] + "_" + df[c_res] + "_" + df[c_mut] + ".pdb")
            have = int(names.isin(muts).sum())
            info["scap_mutant_present"] = have
            info["scap_mutant_missing"] = int(len(df) - have)
            print("      SCAP mutant present: %d / %d (missing %d)" % (have, len(df), len(df) - have))
        out[p.name] = info
    return out


@guarded("3. SCAP / structures on disk")
def check_structures():
    out = {
        "mutants_pdb": count("data/features/mutants/*.pdb"),
        "skempi_pdbs": count("data/raw/skempi/pdbs/*.pdb"),
        "abbind_pdbs": count("data/structures/abbind/*.pdb"),
        "antibody_rbd_pdb": count("data/structures/antibody_rbd/*.pdb"),
        "antibody_rbd_cif": count("data/structures/antibody_rbd/*.cif"),
        "antibody_rbd_converted_pdb": count("data/structures/antibody_rbd_converted/*.pdb"),
        "6M0J": (ROOT / "data/structures/antibody_rbd/6M0J.pdb").is_file(),
        "7KL9": (ROOT / "data/structures/ctc445/7KL9.pdb").is_file(),
    }
    for k, v in out.items():
        print("  %-28s %s" % (k, v))
    log = ROOT / "data/raw/skempi/s1131_for_features.scap_log.csv"
    if log.is_file():
        info, df = csv_info(log)
        c = pick(df.columns, "status", "ok", "result", "error")
        if c is not None:
            print("      %s: %s" % (c, df[c].value_counts().head(8).to_dict()))
    return out


@guarded("4. H0 / aux outputs already generated")
def check_features():
    out = {}
    for d in sorted((ROOT / "data/features").glob("*")):
        if not d.is_dir() or d.name in ("rows", "mutants"):
            continue
        rows = d / "rows"
        item = {"row_files": len(list(rows.glob("*.npy"))) if rows.is_dir() else None}
        for name in ("features_full.npy", "ok_mask.npy"):
            if (d / name).is_file():
                item[name] = npy_info(d / name)
                if name == "ok_mask.npy":
                    m = np.load(d / name)
                    item["ok_true"], item["ok_total"] = int(m.sum()), int(m.size)
        if (d / "run_summary.json").is_file():
            item["run_summary"] = json.loads((d / "run_summary.json").read_text())
        if rows.is_dir() and item["row_files"]:
            first = sorted(rows.glob("*.npy"))[0]
            item["example_row"] = npy_info(first)
            arr = np.load(first)
            item["example_row_nan"] = int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0
        out[d.name] = item
        print("  %s -> %s" % (d.name, json.dumps(item, default=str)[:900]))
    h12 = [str(p.relative_to(ROOT)) for p in (ROOT / "data/features").glob("*h12*")]
    out["h12_dirs_found"] = h12
    print("  H1/H2 output dirs found: %s" % (h12 or "NONE"))
    cmp_json = ROOT / "data/features/compare_h0.json"
    if cmp_json.is_file():
        print("  compare_h0.json (first 1500 chars):\n" + cmp_json.read_text()[:1500])
    return out


@guarded("5. what the repo feature modules expose")
def check_modules():
    sys.path.insert(0, str(ROOT / "src"))
    out = {}
    for name in ("h0", "h12", "aux", "manifest", "pdbio"):
        try:
            mod = importlib.import_module("topnet.features." + name)
        except Exception as exc:  # noqa: BLE001
            out[name] = "IMPORT FAILED: %s: %s" % (type(exc).__name__, exc)
            print("  %-8s %s" % (name, out[name]))
            continue
        funcs = {}
        for fname, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ != mod.__name__ or fname.startswith("_"):
                continue
            doc = (inspect.getdoc(fn) or "").splitlines()
            funcs[fname] = {"sig": str(inspect.signature(fn)), "doc": doc[0] if doc else ""}
        out[name] = funcs
        print("  topnet.features.%s" % name)
        for fname, d in funcs.items():
            print("      %s%s  %s" % (fname, d["sig"], ("-- " + d["doc"]) if d["doc"] else ""))
    return out


@guarded("6. tools on this machine")
def check_tools():
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "sklearn", "torch", "gudhi", "ripser", "Bio", "yaml", "joblib"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "ok")
        except Exception:  # noqa: BLE001
            out[mod] = None
    for exe in ("R", "Rscript", "matlab", "java", "vmd", "scap", "profix", "pdb2pqr", "psiblast", "nvidia-smi"):
        out[exe] = shutil.which(exe)
    if out.get("Rscript"):
        try:
            r = subprocess.run(
                ["Rscript", "-e", 'cat(requireNamespace("TDA", quietly=TRUE))'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60,
            )
            out["R:TDA"] = r.stdout.strip()
        except Exception as exc:  # noqa: BLE001
            out["R:TDA"] = "check failed: %s" % exc
    for k, v in out.items():
        print("  %-12s %s" % (k, v))
    return out


@guarded("7. capsule (shipped reference data)")
def check_capsule(capsule):
    if capsule is None:
        print("  (no --capsule given)")
        return {}
    cap = Path(capsule)
    out = {}
    for rel in (
        "code/crossvalidation/X.npy", "code/crossvalidation/Y.npy", "code/crossvalidation/Y_skempi.npy",
        "code/crossvalidation/X_skempi_feature1.npy", "code/crossvalidation/X_skempi_feature2.npy",
        "code/crossvalidation/crosslabel.npy", "code/crossvalidation/importanceidx.npy",
        "code/src/model/X_unnormed.npy", "code/src/model/X_aux.npy", "code/src/model/importance_idx.npy",
        "data/skempi_1131.npy",
    ):
        p = cap / rel
        out[rel] = npy_info(p) if p.is_file() else None
        print("  %-48s %s" % (rel, out[rel]["shape"] if out[rel] else "MISSING"))
    for rel in ("code/src/gen_feature_topo/gen_feature_12.py", "code/src/gen_feature_topo/PH_Alpha.R",
                "code/src/gen_feature_topo/Gen_pointclouds.py", "code/src/gen_feature_topo/Construct_Feature_tda.py",
                "data/software/jackal_64bit/bin/scap", "data/software/jackal_64bit/bin/profix"):
        out[rel] = (cap / rel).is_file()
        print("  %-48s %s" % (rel, "present" if out[rel] else "MISSING"))
    return out


# ----------------------------------------------------------------- bundles
REPO_FILES = [
    "AGENTS.md", "Makefile", "pyproject.toml", "configs/default.yaml", "configs/tuned.yaml",
    "docs/CAPSULE_PIPELINE.md", "docs/LEGACY_DIFFS.md", "docs/DATA_NOTES.md", "docs/RUNBOOK_NEXT.md",
    "docs/ALIGNMENT_G0.md", "docs/RESULTS.md",
    "src/topnet/features/h0.py", "src/topnet/features/h12.py", "src/topnet/features/aux.py",
    "src/topnet/features/manifest.py", "src/topnet/features/pdbio.py",
    "scripts/36_run_generator.py", "scripts/41_build_feature_rows.py", "scripts/41_make_s645_rows.py",
    "scripts/42_build_scap_mutants.py", "scripts/35_compare_features.py", "scripts/reproduce_h0.py",
    "scripts/g2_one_row.py", "scripts/60_merge_labels.py", "scripts/34_join_s1131.py",
    "tests/test_h0.py", "tests/test_h12_and_aux.py", "tests/test_feature_rows.py",
    "data/features/compare_h0.json", "data/features/s1131_h0/run_summary.json",
    "data/raw/probe_report.json", "data/mutation_sets/target_report.json",
]
REPO_CSV = [
    "data/features/rows/skempi2.csv", "data/features/rows/starr2020.csv", "data/features/rows/chan2020.csv",
    "data/features/rows/panel.csv", "data/features/rows/panel_numbering.csv",
    "data/labels/reconstructed_labels.csv", "data/labels/reconstructed_labels_dropped.csv",
    "data/features/compare_h0_columns.csv", "data/features/s1131_h0/index.csv",
    "data/raw/skempi/s1131_for_features.scap_log.csv", "data/structures/antibody_rbd/complex_map_final.csv",
    "data/predict/rows_26x186.csv", "data/splits/labels_with_folds.csv",
]
REPO_LOGS_TAIL = ["logs/G0_preflight.log", "logs/alignment.log", "logs/alignment_C.log", "logs/s645_scap.log"]

CAPSULE_FILES = [
    "code/src/gen_feature_topo/gen_feature_12.py", "code/src/gen_feature_topo/PH_Alpha.R",
    "code/src/gen_feature_topo/Gen_pointclouds.py", "code/src/gen_feature_topo/Construct_Feature_tda.py",
    "code/src/gen_feature_topo/MutationModule.py", "code/src/gen_feature_topo/modify_pdb.py",
    "code/src/gen_feature_topo/Construct_jplex_binding.m", "code/src/gen_feature_topo/Construct_jplex_mutation.m",
    "code/src/gen_feature_topo/java.opts", "code/src/Gen_feature_topo.py", "code/src/predict.py",
    "code/src/model/Construct_Feature.py", "code/src/model/train_model_cnn.py",
    "code/crossvalidation/crossvalidation_1131_v2.py", "code/crossvalidation/crossvalidation_ab.py",
    "code/readme.txt", "code/note_for_reproducible_run.txt", "code/run_topnettree.sh", "code/pre_setting.sh", "code/run",
]
MAX_BYTES = 150_000


def _append(out, base, rel, mode="full"):
    p = base / rel
    out.write("\n\n" + "#" * 100 + "\n##### %s\n" % rel + "#" * 100 + "\n")
    if not p.is_file():
        out.write("<<MISSING>>\n")
        return
    if mode == "csv":
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        out.write("rows=%d\ncolumns=%s\n" % (len(df), list(df.columns)))
        out.write(df.head(6).to_csv(index=False))
        return
    text = p.read_text(errors="replace")
    if mode == "tail":
        text = "\n".join(text.splitlines()[-60:])
    elif len(text) > MAX_BYTES:
        text = text[:MAX_BYTES] + "\n<<TRUNCATED>>"
    out.write(text + "\n")


def write_bundles(capsule):
    section("8. bundles")
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs/bundle_repo.txt", "w") as fh:
        for rel in REPO_FILES:
            _append(fh, ROOT, rel)
        for rel in REPO_CSV:
            try:
                _append(fh, ROOT, rel, "csv")
            except Exception as exc:  # noqa: BLE001
                fh.write("<<could not read %s: %s>>\n" % (rel, exc))
        for rel in REPO_LOGS_TAIL:
            _append(fh, ROOT, rel, "tail")
    print("  wrote logs/bundle_repo.txt")
    if capsule is not None:
        with open(ROOT / "logs/bundle_capsule.txt", "w") as fh:
            for rel in CAPSULE_FILES:
                _append(fh, Path(capsule), rel)
        print("  wrote logs/bundle_capsule.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capsule", default=None, help="root of the TopNetTree capsule (contains code/ and data/)")
    ap.add_argument("--bundle", action="store_true", help="also write logs/bundle_repo.txt and logs/bundle_capsule.txt")
    args = ap.parse_args()
    print("repo root: %s" % ROOT)
    check_labels()
    check_rows()
    check_structures()
    check_features()
    check_modules()
    check_tools()
    check_capsule(args.capsule)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs/inventory.json").write_text(json.dumps(REPORT, indent=2, default=str))
    print("\nwrote logs/inventory.json")
    if args.bundle:
        write_bundles(args.capsule)


if __name__ == "__main__":
    main()
