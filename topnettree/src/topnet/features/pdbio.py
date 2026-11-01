"""PDB atom reader aligned with Gen_pointclouds.py.

The legacy reader keeps element character ``line[13]``, chain ``line[21]``,
and residue text ``line[23:26]`` (it drops the first character of the 4-column
resSeq). For residue numbers below 1000 that dropped character is a space.
For residue numbers >= 1000 it drops the thousands digit. The default here
keeps the full resSeq plus the insertion code. Pass ``legacy_resid=True`` to
reproduce the old slice.
"""

from __future__ import annotations

from pathlib import Path

ELEMENTS = ("C", "N", "O")


def read_pdb(path: str | Path, legacy_resid: bool = False) -> list[tuple]:
    atoms = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM") or len(line) < 54:
                continue
            element = line[13:14]
            if element not in ELEMENTS:
                continue
            chain = line[21:22]
            if legacy_resid:
                resid = line[23:26].strip()
            else:
                resid = line[22:26].strip()
                insertion = line[26:27].strip() if len(line) > 26 else ""
                if insertion and not insertion.isdigit():
                    resid = f"{resid}{insertion}"
            atoms.append(
                (
                    chain,
                    resid,
                    element,
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    return atoms
