from topnet.features.pdbio import is_hydrogen, read_pdb_records


def _line(serial, name4, element=""):
    return (f"ATOM  {serial:5d} {name4:4s} LYS A{serial:4d}    "
            f"{1.0:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}")


def test_scap_style_hydrogen_names(tmp_path):
    names = [(" CA ", "", False), (" HA ", "", True), ("1HG1", "", True), ("2HE ", "", True), ("1HZ ", "H", True), (" NZ ", "N", False)]
    p = tmp_path / "x.pdb"
    p.write_text("\n".join(_line(i + 1, n, e) for i, (n, e, _) in enumerate(names)) + "\n")
    recs = read_pdb_records(p, altloc="all")
    assert [is_hydrogen(r) for r in recs] == [h for _, _, h in names]