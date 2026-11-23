#!/usr/bin/env python3
"""Does every built mutant differ from its input structure at exactly the requested residue?

Residues are matched by position inside each chain, so renumbering by SCAP does not matter.
Status: ok | renumbered (right residue, numbers or insertion codes differ from the input) | wrong_site |
wrong_residue | structure_changed | missing | bad_row.

    python scripts/verify_site_mutants.py --rows data/raw/s645_for_features.csv --include-renumbered --out-bad data/raw/s645_rebuild.csv
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from topnet.features.pdbio import AA1  # noqa: E402

RESID = re.compile(r"^(-?\d+)([A-Za-z]?)$")


def residues(path):
    out, last = {}, None
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.startswith("ATOM") or len(ln) < 27:
            continue
        k = (ln[21], ln[22:26].strip(), ln[26].strip().upper())
        if k != last:
            out.setdefault(k[0], []).append((k[1], k[2], ln[17:20].strip()))
            last = k
    return out


def verify(rec):
    mut = ROOT / rec["mutant"]
    if not mut.is_file() or mut.stat().st_size < 1000:
        return "missing", ""
    src = Path(rec["structure"])
    if not src.is_file():
        src = ROOT / rec["structure"]
    m = RESID.match(rec["resid_struct"].strip())
    target = rec["mut"].strip().upper()
    if not src.is_file() or not m or target not in AA1:
        return "bad_row", f"{rec['structure']} {rec['resid_struct']} {rec['mut']}"
    want = (m.group(1), m.group(2).upper())
    a, b = residues(src), residues(mut)
    if set(a) != set(b) or any(len(a[c]) != len(b[c]) for c in a):
        return "structure_changed", ("residues per chain in/out: " + ", ".join(f"{c}:{len(a[c])}/{len(b.get(c, []))}" for c in sorted(a)))[:150]
    pos = next((i for i, (n, ic, _) in enumerate(a.get(rec["chain"], [])) if (n, ic) == want), None)
    if pos is None:
        return "bad_row", f"site {rec['resid_struct']} not in input chain {rec['chain']}"
    diff = [(c, i) for c in a for i, (x, y) in enumerate(zip(a[c], b[c])) if x[2] != y[2]]
    if diff != [(rec["chain"], pos)]:
        where = ", ".join(f"{c}:{a[c][i][0]}{a[c][i][1]} {a[c][i][2]}->{b[c][i][2]}" for c, i in diff[:4]) or "nothing changed"
        return "wrong_site", f"wanted {rec['chain']}:{rec['resid_struct']}; changed {where}"
    if b[rec["chain"]][pos][2] != AA1[target]:
        return "wrong_residue", f"built {b[rec['chain']][pos][2]}, wanted {AA1[target]}"
    if any(x[:2] != y[:2] for c in a for x, y in zip(a[c], b[c])):
        return "renumbered", "residue numbers differ from the input"
    return "ok", ""


ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--where", action="append", default=[], metavar="COL=VAL")
ap.add_argument("--out-bad", help="write the rows that need a rebuild here")
ap.add_argument("--include-renumbered", action="store_true", help="also rebuild rows whose only problem is numbering")
args = ap.parse_args()
rows = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
for cond in args.where:
    col, val = cond.split("=", 1)
    rows = rows[rows[col] == val]
rows = rows.drop_duplicates("mutant").reset_index(drop=True)
res = [verify(r) for r in rows.to_dict("records")]
rows["verify"], rows["detail"] = [s for s, _ in res], [d for _, d in res]
print(f"{len(rows)} distinct mutants: {dict(Counter(rows['verify']))}")
for _, r in rows[~rows["verify"].isin(["ok", "renumbered", "missing"])].head(15).iterrows():
    print(f"  {r['verify']:17} {r['mutant']}: {r['detail']}")
if args.out_bad:
    bad = {"missing", "wrong_site", "wrong_residue", "structure_changed"} | ({"renumbered"} if args.include_renumbered else set())
    out = rows[rows["verify"].isin(bad)].drop(columns=["verify", "detail"])
    out.to_csv(args.out_bad, index=False)
    print(f"wrote {args.out_bad}: {len(out)} rows to rebuild")