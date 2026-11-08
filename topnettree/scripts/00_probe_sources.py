#!/usr/bin/env python3
"""Step 0. Look before building. Prints the real structure of every source the other scripts depend on.

Nothing here downloads a full archive. It reads headers, the first members of a tarball, and a few RCSB
records, then writes data/raw/probe_report.json. Paste the printed summary back if anything looks off.

    python scripts/00_probe_sources.py [--root .]
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import UA, get_json, is_lfs_pointer, looks_like_html, try_get  # noqa: E402

SKEMPI_CSV = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
SKEMPI_TGZ = "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz"
STARR_RAW = "https://raw.githubusercontent.com/jbloomlab/SARS-CoV-2-RBD_DMS/{branch}/{path}"
STARR_MEDIA = "https://media.githubusercontent.com/media/jbloomlab/SARS-CoV-2-RBD_DMS/{branch}/{path}"
STARR_MAIN = "results/single_mut_effects/single_mut_effects.csv"
STARR_MAVEDB = "MaveDB_upload/score-set_binding_mut.csv"
STARR_SITES = "data/RBD_sites.csv"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pid}"
RCSB_ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pid}/{eid}"
RCSB_FILE = "https://files.rcsb.org/header/{pid}.pdb"
HOSTS = [
    "https://life.bsc.es/pid/skempi2/",  # the site root times out (504); the SKEMPI pages and downloads respond
    "https://files.rcsb.org/",
    "https://data.rcsb.org/",
    "https://github.com/",
    "https://raw.githubusercontent.com/",
    "https://media.githubusercontent.com/",
]
RCSB_PROBE_IDS = ["6XDG", "7L7E", "7L7D", "6M0J", "7KMG", "1061", "7T9L"]


def section(title: str) -> None:
    print(f"\n=== {title}")


def probe_local(root: Path) -> dict:
    out: dict = {"root": str(root)}
    names = root / "data/raw/antibodies.csv"
    bfes = root / "data/raw/antibodies_BFEs.csv"
    out["antibodies_csv"] = names.is_file()
    out["antibodies_bfes_csv"] = bfes.is_file()
    if names.is_file() and bfes.is_file():
        import pandas as pd

        labels = [x.rstrip("\n") for x in names.open()]
        bf = pd.read_csv(bfes)
        out["n_labels"] = len(labels)
        out["bfe_shape"] = list(bf.shape)
        out["labels_match_bfe_header"] = [x.split("_", 1)[1] for x in labels] == list(bf.columns)
        out["ace2_sum"] = float(bf["ACE2"].sum())
        print(f"antibodies.csv: {len(labels)} labels | BFE table {bf.shape} | header matches: {out['labels_match_bfe_header']}"
              f" | ACE2 column sum {out['ace2_sum']:.4f}")
    import numpy as np

    found = {}
    for pat in ("X_skempi_feature1.npy", "X_skempi_feature2.npy", "Y_skempi.npy", "skempi_1131.npy"):
        for hit in list(root.rglob(pat))[:3]:
            try:
                arr = np.load(hit, mmap_mode="r", allow_pickle=True)
                found[str(hit.relative_to(root))] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            except Exception as err:
                found[str(hit.relative_to(root))] = {"error": str(err)[:120]}
    out["skempi_1131_files"] = found
    print("precomputed SKEMPI-1131 files:", found or "none found under root (copy the capsule files in, or ignore)")
    dms = root / "data/raw/dms"
    out["dms_dir_files"] = sorted(str(p.relative_to(root)) for p in dms.rglob("*") if p.is_file()) if dms.is_dir() else []
    print("data/raw/dms contents:", out["dms_dir_files"] or "empty (Chan 2020 / Linsky 2020 files go here)")
    return out


def probe_hosts() -> dict:
    out = {}
    for url in HOSTS:
        data, status = try_get(url, timeout=10, retries=1, byte_range=(0, 200))
        out[url] = status
        print(f"{url:45s} {status}")
    return out


def probe_skempi() -> dict:
    out: dict = {}
    data, status = try_get(SKEMPI_CSV, timeout=30, retries=1, byte_range=(0, 8000))
    out["csv_status"] = status
    if data is None:
        print("skempi_v2.csv unreachable:", status)
    elif looks_like_html(data):
        out["csv_status"] = "html (not a csv)"
        print("skempi_v2.csv returned HTML. Download it in a browser and pass --csv-local to 10_fetch_skempi.py")
    else:
        lines = data.decode("utf-8", "replace").splitlines()
        delim = ";" if lines[0].count(";") > lines[0].count(",") else ","
        cols = lines[0].split(delim)
        out.update({"delimiter": delim, "columns": cols, "first_row": lines[1].split(delim) if len(lines) > 1 else None})
        print(f"skempi csv delimiter {delim!r}, {len(cols)} columns:\n  {cols}")
        print("  first row:", out["first_row"])
    # stream only the first members of the tarball
    try:
        req = urllib.request.Request(SKEMPI_TGZ, headers=UA)
        names = []
        with urllib.request.urlopen(req, timeout=30) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for member in tar:
                    names.append(member.name)
                    if len(names) >= 12:
                        break
        out["tgz_first_members"] = names
        print("SKEMPI2_PDBs.tgz first members:", names)
    except Exception as err:
        out["tgz_first_members"] = f"unreadable: {type(err).__name__}: {err}"[:200]
        print("SKEMPI2_PDBs.tgz not readable:", out["tgz_first_members"])
    return out


def probe_starr() -> dict:
    out: dict = {}
    for label, path in (("main", STARR_MAIN), ("mavedb", STARR_MAVEDB), ("sites", STARR_SITES)):
        rec: dict = {}
        for branch in ("master", "main"):
            data, status = try_get(STARR_RAW.format(branch=branch, path=path), timeout=30, retries=1, byte_range=(0, 1500))
            if data is None:
                rec[f"raw_{branch}"] = status
                continue
            rec["branch"] = branch
            if is_lfs_pointer(data):
                rec["raw"] = "git-lfs pointer, not data"
                media, mstatus = try_get(STARR_MEDIA.format(branch=branch, path=path), timeout=30, retries=1, byte_range=(0, 1500))
                rec["media"] = mstatus
                if media is not None and not is_lfs_pointer(media):
                    head = media.decode("utf-8", "replace").splitlines()
                    rec["header"] = head[0]
                    rec["first_row"] = head[1] if len(head) > 1 else None
            else:
                head = data.decode("utf-8", "replace").splitlines()
                rec["raw"] = "data"
                rec["header"] = head[0]
                rec["first_row"] = head[1] if len(head) > 1 else None
            break
        out[label] = rec
        print(f"starr {label}: {rec}")
    return out


def probe_rcsb() -> dict:
    out: dict = {}
    for pid in RCSB_PROBE_IDS:
        entry, status = get_json(RCSB_ENTRY.format(pid=pid))
        if entry is None:
            out[pid] = {"status": status}
            print(f"RCSB {pid}: {status}")
            continue
        info = {"title": (entry.get("struct") or {}).get("title"), "entities": []}
        ids = (entry.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []
        for eid in ids:
            ent, _ = get_json(RCSB_ENTITY.format(pid=pid, eid=eid))
            if ent is None:
                continue
            info["entities"].append(
                {
                    "entity": eid,
                    "description": (ent.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
                    "auth_chains": (ent.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids"),
                }
            )
        out[pid] = info
        print(f"RCSB {pid}: {info['title']}")
        for e in info["entities"]:
            print(f"    entity {e['entity']}: {e['description']}  chains {e['auth_chains']}")
    data, status = try_get(RCSB_FILE.format(pid="6M0J"), timeout=20, retries=1, byte_range=(0, 300))
    out["files.rcsb.org"] = status
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="repository root (the folder that contains data/)")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report: dict = {}
    for name, fn in (
        ("local", lambda: probe_local(root)),
        ("hosts", probe_hosts),
        ("skempi", probe_skempi),
        ("starr_dms", probe_starr),
        ("rcsb", probe_rcsb),
    ):
        section(name)
        try:
            report[name] = fn()
        except Exception as err:  # the probe must never stop halfway
            report[name] = {"probe_error": f"{type(err).__name__}: {err}"}
            print("probe error:", report[name]["probe_error"])
    section("Chan 2020 / Linsky 2020")
    print("No verified machine-readable URL is hard-coded for either dataset. Put the supplementary tables from the\n"
          "Science papers in data/raw/dms/chan2020/ and data/raw/dms/linsky2020/; 20_fetch_dms.py prints their columns\n"
          "before it reads anything.")
    out = root / "data/raw/probe_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
