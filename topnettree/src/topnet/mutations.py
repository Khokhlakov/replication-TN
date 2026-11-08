"""RBD mutation sets used by the paper. Everything here is read from the text and Fig. 2 caption of the JPCL letter,
except the earlier-variant definitions, which are the standard RBD-only lineage definitions (not stated in the paper)."""

from __future__ import annotations

import re

SHARED12 = ["G339D", "S373P", "S375F", "K417N", "N440K", "S477N", "T478K", "E484A", "Q493R", "Q498R", "N501Y", "Y505H"]
BA1_ONLY = ["S371L", "G446S", "G496S"]
BA2_ONLY = ["S371F", "T376A", "D405N", "R408S"]
BA3_ONLY = ["S371F", "D405N", "G446S"]  # all three also occur in BA.1 or BA.2
EARLIER7 = ["R346K", "K417T", "L452R", "L452Q", "E484Q", "E484K", "F490S"]

# Fig. 2 row order: 7 earlier, 12 shared, 3 BA.1, 4 BA.2 (26 rows)
FIG2_ORDER = EARLIER7 + SHARED12 + BA1_ONLY + BA2_ONLY
# Row order of the shipped antibodies_BFEs.csv (15 BA.1 mutations, sorted by residue number)
BA1_ROW_ORDER = ["G339D", "S371L", "S373P", "S375F", "K417N", "N440K", "G446S", "S477N", "T478K", "E484A",
                 "Q493R", "G496S", "Q498R", "N501Y", "Y505H"]
MISSING_11 = [m for m in FIG2_ORDER if m not in BA1_ROW_ORDER]

VARIANTS = {
    "BA.1": SHARED12 + BA1_ONLY,
    "BA.2": SHARED12 + BA2_ONLY,
    "BA.3": SHARED12 + BA3_ONLY,
    "Delta": ["L452R", "T478K"],
    "Alpha": ["N501Y"],
    "Beta": ["K417N", "E484K", "N501Y"],
    "Gamma": ["K417T", "E484K", "N501Y"],
    "Theta": ["K417T", "E484K", "N501Y"],
    "Kappa": ["L452R", "E484Q"],
    "Lambda": ["L452Q", "F490S"],
    "Mu": ["R346K", "E484K", "N501Y"],
}


def split(mutation: str):
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mutation)
    return m.group(1), int(m.group(2)), m.group(3)


assert len(FIG2_ORDER) == 26 and len(set(FIG2_ORDER)) == 26
assert len(MISSING_11) == 11 and set(BA1_ROW_ORDER) == set(SHARED12 + BA1_ONLY)
