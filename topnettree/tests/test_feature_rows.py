"""Panel paths come from complex_map_final.csv, including the five converted PDBs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("rows41", ROOT / "scripts" / "41_build_feature_rows.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panel_uses_file_column_and_paper_numbering():
    builder = load_builder()
    panel, audit = builder.panel_rows(ROOT / "data/structures/antibody_rbd/complex_map_final.csv")
    assert len(panel) == 26 * 186
    converted = audit[audit["file"].str.contains("antibody_rbd_converted")]
    assert sorted(converted["resolved_id"]) == ["6ZBP", "7B27", "7OAO", "7OAP", "7OAY"]
    assert set(panel.loc[panel["pdb"].isin(converted["resolved_id"]), "structure"]) == set(converted["file"])
    assert (converted["offset"] == 0).all()
    assert (converted["wt_matches"] == 26).all()
    assert int(panel.loc[panel["pdb"].isin(converted["resolved_id"]), "wt_ok"].sum()) == 26 * len(converted)
    b27 = panel[panel["pdb"] == "7B27"]
    assert (b27["resid_struct"] == b27["resid"]).all()
    assert (b27["structure"] == "data/structures/antibody_rbd_converted/7B27.pdb").all()


def test_s1131_rows_keep_joined_order_and_resid_struct():
    builder = load_builder()
    rows = builder.s1131_rows(
        ROOT / "data/raw/skempi/s1131_joined.csv",
        ROOT / "data/raw/skempi/skempi_v2_single_labels.csv",
    )
    joined = pd.read_csv(ROOT / "data/raw/skempi/s1131_joined.csv", dtype=str, keep_default_na=False)
    assert len(rows) == len(joined) == 1126
    assert rows["s1131_row"].tolist() == joined["s1131_row"].tolist()
    assert rows["resid_struct"].ne("").all()
    assert rows["structure"].str.startswith("data/raw/skempi/pdbs/").all()


def test_bfe_header_matches_library_labels():
    builder = load_builder()
    _panel, audit = builder.panel_rows(ROOT / "data/structures/antibody_rbd/complex_map_final.csv")
    header = pd.read_csv(ROOT / "data/raw/antibodies_BFEs.csv", nrows=0).columns.tolist()
    assert audit["label"].tolist() == header
