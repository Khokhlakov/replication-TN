"""Geometry of the capsule homology blocks on a hand-built two-chain PDB.

The coordinates are a fixture for the neighborhood cut and the bin layout.
They are not a stand-in for an AB-Bind row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from topnet.features.h0 import quantize_filtration
from topnet.features.h12 import H12_DIM, alpha_persistence
from topnet.features.homology import (
    BIND_DEATH,
    DIFF_PAIRS,
    MUT_DEATH,
    homology_blocks,
    legacy_residue_key,
    parse_site,
)
from topnet.features.pdbio import read_pdb_records


def pdb_line(serial: int, element: str, chain: str, resseq: int, x: float, y: float, z: float) -> str:
    # PDB column 14 is the element letter. That is the second character of the atom name.
    name = f" {element}  "[:4]
    return (
        f"ATOM  {serial:5d} {name} ALA {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}\n"
    )


def write_complex(path: Path, partner_distance: float) -> None:
    """Chain A residue 10, one carbon. Chain B, one nitrogen at ``partner_distance`` and one at 40 A."""
    text = "".join(
        [
            pdb_line(1, "C", "A", 10, 0.0, 0.0, 0.0),
            pdb_line(2, "N", "B", 3, partner_distance, 0.0, 0.0),
            pdb_line(3, "N", "B", 4, 40.0, 0.0, 0.0),
            "END\n",
        ]
    )
    path.write_text(text, encoding="utf-8")


def test_reader_and_legacy_key_on_the_fixture(tmp_path: Path):
    path = tmp_path / "wild.pdb"
    write_complex(path, 5.0)
    records = read_pdb_records(path)
    assert [rec["element"] for rec in records] == ["C", "N", "N"]
    assert records[0]["resseq_int"] == 10
    assert records[0]["xyz"] == (0.0, 0.0, 0.0)
    assert legacy_residue_key(1001) == "001"
    assert parse_site("100A") == (100, "A")
    assert parse_site(100, "A") == (100, "A")


def test_quantize_grid_matches_the_stream_divisions():
    snapped = quantize_filtration(np.array([0.075, 0.08, 15.0]))
    assert np.allclose(snapped, [0.075, 0.15, 15.0])


def test_binding_death_uses_the_near_partner_only(tmp_path: Path):
    wild = tmp_path / "wild.pdb"
    write_complex(wild, 5.0)
    blocks = homology_blocks(wild, wild, chain="A", resid=10, compute_h12=False)
    assert blocks.h0.shape == (48, 108)
    assert blocks.h12 is None
    # One cross pair, C on A and N on B, 5.0 A. The boundary rule puts 5.0 in bin 20.
    # Pair CN is column 1. The nitrogen at 40 A must not add a second bar.
    death = blocks.h0[:, BIND_DEATH]
    assert death[20, 1] == 1
    assert death[:, 1].sum() == 1
    assert np.all(blocks.h0[:, BIND_DEATH][:, DIFF_PAIRS] == 0)
    assert np.all(blocks.h0[:, MUT_DEATH][:, DIFF_PAIRS] == 0)


def test_moving_the_partner_changes_the_wild_minus_mutant_columns(tmp_path: Path):
    wild = tmp_path / "wild.pdb"
    mutant = tmp_path / "mutant.pdb"
    write_complex(wild, 5.0)
    write_complex(mutant, 6.0)
    blocks = homology_blocks(wild, mutant, chain="A", resid=10, compute_h12=False, quantize=False)
    diff = blocks.h0[:, BIND_DEATH][:, DIFF_PAIRS]
    assert diff[20, 1] == 1
    assert diff[24, 1] == -1


def test_missing_residue_raises(tmp_path: Path):
    wild = tmp_path / "wild.pdb"
    write_complex(wild, 5.0)
    with pytest.raises(ValueError, match="No homology block"):
        homology_blocks(wild, wild, chain="A", resid=99, compute_h12=False)


def test_legacy_key_selects_residue_1001(tmp_path: Path):
    path = tmp_path / "wild.pdb"
    text = "".join(
        [
            pdb_line(1, "C", "A", 1001, 0.0, 0.0, 0.0),
            pdb_line(2, "N", "B", 3, 5.0, 0.0, 0.0),
            pdb_line(3, "C", "A", 1, 0.0, 0.0, 25.0),
            "END\n",
        ]
    )
    path.write_text(text, encoding="utf-8")
    blocks = homology_blocks(path, path, chain="A", resid=1001, legacy_resid=True, compute_h12=False)
    death = blocks.h0[:, MUT_DEATH]
    assert death[20, 1] == 1
    assert death[:, 1].sum() == 1


def test_square_has_an_h1_bar():
    pytest.importorskip("gudhi")
    square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    births, deaths = alpha_persistence(square)[1]
    assert len(births) >= 1
    assert np.all(deaths + 1e-8 >= births)


def test_h12_difference_is_zero_when_the_mutant_matches(tmp_path: Path):
    pytest.importorskip("gudhi")
    wild = tmp_path / "wild.pdb"
    write_complex(wild, 5.0)
    blocks = homology_blocks(wild, wild, chain="A", resid=10)
    assert blocks.h12 is not None
    assert blocks.h12.shape == (H12_DIM,)
    assert np.all(blocks.h12.reshape(-1, 3)[:, 2] == 0)


def test_script_writes_h0_only(tmp_path: Path, script_loader):
    wild = tmp_path / "wild.pdb"
    out = tmp_path / "out"
    write_complex(wild, 5.0)
    script = script_loader("write_homology_blocks")
    code = script.main(
        ["--wild", str(wild), "--mutant", str(wild), "--chain", "A", "--resid", "10", "--h0-only", "--out-dir", str(out)]
    )
    assert code == 0
    assert np.load(out / "h0.npy").shape == (48, 108)
    assert not (out / "h12.npy").exists()


def test_script_does_not_invent_h12_when_gudhi_is_missing(tmp_path: Path, script_loader, monkeypatch):
    wild = tmp_path / "wild.pdb"
    out = tmp_path / "out"
    write_complex(wild, 5.0)
    script = script_loader("write_homology_blocks")
    monkeypatch.setattr(script, "gudhi_available", lambda: False)
    code = script.main(
        ["--wild", str(wild), "--mutant", str(wild), "--chain", "A", "--resid", "10", "--out-dir", str(out)]
    )
    assert code == 2
    assert np.load(out / "h0.npy").shape == (48, 108)
    assert not (out / "h12.npy").exists()


def test_script_writes_nothing_when_the_residue_is_missing(tmp_path: Path, script_loader):
    wild = tmp_path / "wild.pdb"
    out = tmp_path / "out"
    write_complex(wild, 5.0)
    script = script_loader("write_homology_blocks")
    code = script.main(
        ["--wild", str(wild), "--mutant", str(wild), "--chain", "A", "--resid", "99", "--h0-only", "--out-dir", str(out)]
    )
    assert code == 1
    assert not out.exists()
