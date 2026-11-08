#!/usr/bin/env python3
"""Step 7. Freeze the reconstructed data: sha256 manifest plus an auto-filled DATA_CARD.md.

Hashes every file under the listed data folders (structure folders are hashed file by file too), counts rows/columns of every CSV,
and embeds the machine-written reports from earlier steps (numbering_check, s1131_join_report, compare_*.json, run_summary.json,
target_report.json, complex_map_final QC). The 'Deviations' section is read from data/deviations.md if it exists; edit that file
by hand, it is the place to write every known difference from the paper.

    python scripts/70_data_card.py [--root data]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPORTS = ["raw/skempi/numbering_check.json", "raw/skempi/s1131_join_report.json", "mutation_sets/target_report.json",
           "mutation_sets/ace2_constraints.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_shape(path: Path):
    try:
        with path.open("rb") as fh:
            n = sum(1 for _ in fh) - 1
        ncol = len(pd.read_csv(path, nrows=0, sep=None, engine="python").columns)
        return n, ncol
    except Exception:
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--exclude", nargs="*", default=["__pycache__", ".DS_Store", "/rows/"], help="substrings to skip (per-row feature caches)")
    args = ap.parse_args()
    root = Path(args.root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not any(x in str(p) for x in args.exclude)
                   and p.name not in ("MANIFEST.sha256", "DATA_CARD.md"))
    lines, table = [], []
    for p in files:
        digest = sha256(p)
        lines.append(f"{digest}  {p.as_posix()}")
        n, ncol = csv_shape(p) if p.suffix == ".csv" else (None, None)
        table.append((p.relative_to(root).as_posix(), p.stat().st_size, n, ncol, digest[:12]))
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    card = [f"# Data card\n\nGenerated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from `{root}`; {len(files)} files hashed "
            f"(see MANIFEST.sha256; verify with `sha256sum -c`).\n", "## Files\n", "| file | bytes | rows | cols | sha256[:12] |", "|---|---|---|---|---|"]
    for f, size, n, ncol, d in table:
        if f.endswith(".csv") or size > 5_000_000:
            card.append(f"| {f} | {size} | {n if n is not None else ''} | {ncol if ncol is not None else ''} | {d} |")
    folders = {}
    for f, *_ in table:
        folders.setdefault(f.rsplit("/", 1)[0] if "/" in f else ".", 0)
        folders[f.rsplit("/", 1)[0] if "/" in f else "."] += 1
    card.append("\n## File counts per folder\n")
    card += [f"- `{k}`: {v}" for k, v in sorted(folders.items())]

    card.append("\n## Machine-written reports\n")
    for rel in REPORTS + [str(p.relative_to(root)) for p in sorted(root.rglob("compare_*.json")) + sorted(root.rglob("run_summary.json"))]:
        p = root / rel
        if p.is_file():
            card.append(f"### {rel}\n```json\n{p.read_text()[:3000]}\n```")
    qc = root / "structures/antibody_rbd/complex_map_final.csv"
    if qc.is_file():
        m = pd.read_csv(qc)
        card.append(f"### complex_map_final.csv\n- rows {len(m)}, QC pass {int(m.qc_pass.sum())}, rules {m.rule.value_counts().to_dict()}")
    dev = root / "deviations.md"
    card.append("\n## Deviations from the paper\n")
    card.append(dev.read_text() if dev.is_file() else "_Write data/deviations.md: training-set composition, missing Linsky data, proxy DMS units, "
                "H1/H2 availability, S4169 filter choice, every row dropped and why._")
    (root / "DATA_CARD.md").write_text("\n".join(card) + "\n")
    print(f"wrote {root}/MANIFEST.sha256 ({len(files)} files) and {root}/DATA_CARD.md")


if __name__ == "__main__":
    main()
