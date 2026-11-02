"""T5a model-free checks. CPU. Prints PASS or FAIL for each paper relation.

antibodies_BFEs.csv uses the paper sign: positive means binding strengthened.
AB-Bind S645 is not used here. S645 numbers are not paper results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from topnet.analysis.layer_a import THRESH, disrupted_counts, fold_reduction
from topnet.config import load_config, resolve_path
from topnet.data import assert_label_tables

BA1 = [
    "G339D",
    "S371L",
    "S373P",
    "S375F",
    "K417N",
    "N440K",
    "G446S",
    "S477N",
    "T478K",
    "E484A",
    "Q493R",
    "G496S",
    "Q498R",
    "N501Y",
    "Y505H",
]


def check(name: str, ok: bool, detail: str) -> bool:
    print(("PASS" if ok else "FAIL") + f"  {name}: {detail}")
    return ok


def main() -> int:
    cfg = load_config("configs/default.yaml")
    assert_label_tables(cfg)
    failed = 0

    ratio_21 = float(np.exp(2.98 - 2.60))
    failed += not check(
        "BA.2/BA.1 infectivity exp(2.98-2.60)",
        abs(ratio_21 - 1.5) / 1.5 < 0.05,
        f"computed {ratio_21:.4f}, paper about 1.5",
    )
    ratio_20 = float(np.exp(2.98))
    failed += not check(
        "BA.2/original exp(2.98)",
        abs(ratio_20 - 20.0) / 20.0 < 0.05,
        f"computed {ratio_20:.4f}, paper about 20. Implied original ACE2 sum is 0.",
    )
    delta_bfe = float(np.log(4.2))
    implied = float(np.exp(1.55))
    failed += not check(
        "Delta fold 4.2x versus implied BFE 1.55",
        abs(implied - 4.2) / 4.2 < 0.05,
        f"ln(4.2)={delta_bfe:.4f}, while exp(1.55)={implied:.3f}. "
        "The two prompt figures do not match at 5%. Provenance: prompt arithmetic, not a table in this repo.",
    )

    ba1, ba2, ba3, delta = 20.9, 27.1, 18.1, 1.5
    failed += not check(
        "BA.2/BA.1 escape fold",
        abs(ba2 / ba1 - 1.30) < 0.02,
        f"{ba2}/{ba1}={ba2 / ba1:.4f}, paper about 30% (1.30). Delta fold {delta} is a separate reported value.",
    )
    rates = {
        "BA.1": 163 / 185,
        "BA.2": 168 / 185,
        "BA.3": 164 / 185,
    }
    failed += not check(
        "breakthrough fractions",
        abs(rates["BA.1"] - 0.88) < 0.01 and abs(rates["BA.2"] - 0.91) < 0.01 and abs(rates["BA.3"] - 0.89) < 0.01,
        f"163/185={rates['BA.1']:.4f}, 168/185={rates['BA.2']:.4f}, 164/185={rates['BA.3']:.4f}. Delta count 70/185={70 / 185:.4f}.",
    )
    fig3e = np.array([1.3, 4.2, 4.3, 1.5, 2.5, 2.3, 20.9], dtype=np.float64)
    fig3f = np.array([1.2, 2.8, 1.6, 1.6, 1.7, 4.5, 8.4], dtype=np.float64)
    pearson = float(np.corrcoef(fig3e, fig3f)[0, 1])
    failed += not check(
        "Fig 3e vs Fig 3f Pearson",
        abs(pearson - 0.9) < 0.05,
        f"computed {pearson:.4f}, paper about 0.9. Values are the ones named in the prompt, not re-read from the PDF.",
    )

    frame = pd.read_csv(resolve_path(cfg, "antibodies_bfes"))
    names = (root_lines(resolve_path(cfg, "antibodies")))
    header = list(frame.columns)
    print(f"INFO  antibodies_BFEs.csv shape {frame.shape}")
    suffix = [line.split("_", 1)[-1] for line in names]
    failed += not check(
        "header equals antibodies.csv",
        header == names,
        f"exact equality {header == names}. Equality of the name after the first underscore: {header == suffix}. "
        f"First csv entry {names[0]!r}, first BFE column {header[0]!r}.",
    )
    print("INFO  H15 hypothesis: row order " + " ".join(BA1) + ". Tested with anchors, not assumed.")

    ace = float(frame["ACE2"].sum()) if "ACE2" in frame.columns else float("nan")
    anchor_ace = check("ACE2 sum of 15 rows", abs(ace - 2.60) < 0.05, f"sum {ace:.6f}, Fig 1g BA.1 about 2.60")
    failed += not anchor_ace

    def cell(row_number: int, column: str):
        if column not in frame.columns:
            return None
        return float(frame.iloc[row_number - 1][column])

    k417 = cell(5, "LY-CoV016")
    e484 = cell(10, "LY-CoV555")
    q493 = cell(11, "LY-CoV555")
    g446_cols = [column for column in frame.columns if "REGN10987" in str(column)]
    g446 = {column: cell(7, column) for column in g446_cols}
    anchors_ok = (
        k417 is not None
        and k417 < -1.5
        and e484 is not None
        and e484 < 0
        and q493 is not None
        and q493 < 0
        and any(value is not None and value < 0 for value in g446.values())
    )
    failed += not check(
        "Fig 4 anchors under H15 row numbers",
        anchors_ok,
        f"K417N row5 LY-CoV016={k417}, E484A row10 LY-CoV555={e484}, "
        f"Q493R row11 LY-CoV555={q493}, G446S row7 REGN10987 columns {g446}.",
    )
    if anchors_ok:
        antibodies = frame.drop(columns=["ACE2"]).to_numpy(dtype=np.float64)
        fold = fold_reduction(antibodies)
        clipped_sum = np.clip(antibodies, None, 0.0).sum(axis=0)
        unclipped_sum = antibodies.sum(axis=0)
        n_unclipped = int((unclipped_sum < -0.3).sum())
        n_min = disrupted_counts(antibodies)
        failed += not check(
            "BA.1 fold from clipped mutation sums",
            abs(fold - 20.9) < 0.15,
            f"exp(-mean of per-antibody sum of min(BFE, 0)) = {fold:.4f}, paper 20.9. "
            "Provenance: this file, paper sign. Not an S645 result.",
        )
        print(
            f"INFO  antibodies with unclipped 15-mutation sum < -0.3: {n_unclipped} (paper breakthrough count 163). "
            f"Antibodies whose most negative single mutation is < -0.3: {n_min[1]}."
        )
        print(
            "INFO  strict most-negative counts at thresholds "
            + ", ".join(f"{t:g}:{c}" for t, c in zip(THRESH, n_min))
            + ". Fig 3a2's other drawn thresholds were not re-read from the figure in this run."
        )
        print(
            f"INFO  antibodies with clipped sum < -0.3: {int((clipped_sum < -0.3).sum())}."
        )
    else:
        print("FAIL  anchors failed. No permutation search was run. BA.1 recomputation stopped.")

    pdb_ids = []
    invalid = []
    for line in names:
        token = line.split("_", 1)[0]
        if len(token) == 4 and token.isalnum() and token[0].isdigit():
            pdb_ids.append(token.upper())
        else:
            invalid.append(line)
    print(f"INFO  antibodies.csv entries without a 4-character PDB id: {invalid}")
    print(f"INFO  unique 4-character PDB ids that would need downloading: {len(set(pdb_ids))}")
    print("INFO  BA.2, BA.3, Delta and earlier variants are not rows of antibodies_BFEs.csv. The SI ZIP has only these two files.")
    print(f"INFO  failed checks: {failed}")
    return 1 if failed else 0


def root_lines(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
