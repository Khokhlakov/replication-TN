"""Loaders for the S645 tensors, the reference importance permutation, and labels.

Row order is canonical. Nothing in this module sorts or shuffles samples.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from topnet.config import REQUIRED_LABEL_COLUMNS, load_config, resolve_path

N_SAMPLES = 645
N_BINS = 48
N_CHANNELS = 108
N_AUX = 707
N_CNN_FEATURES = N_BINS * 32  # Keras flatten of (48, 32)

MUTATION_RE = re.compile(r"^([A-Za-z0-9]+):([A-Z])(\d+[A-Za-z]?)([A-Z])$")

# AB-Bind writes non-binders as +8 kcal/mol (affinity lost), matching Y.npy.
NONBINDER_SENTINEL = 8.0


def _default_config():
    return load_config("configs/default.yaml")


def array_paths(cfg: dict | None = None) -> dict[str, Path]:
    cfg = cfg or _default_config()
    return {
        "x_cnn": resolve_path(cfg, "x_cnn"),
        "x_aux": resolve_path(cfg, "x_aux"),
        "importance_idx": resolve_path(cfg, "importance_idx"),
        "labels": resolve_path(cfg, "labels"),
    }


def load_raw_cnn(path: Path | None = None) -> np.ndarray:
    path = path or array_paths()["x_cnn"]
    arr = np.load(path)
    if arr.shape != (N_SAMPLES, N_BINS, N_CHANNELS):
        raise ValueError(f"{path} shape {arr.shape}, expected {(N_SAMPLES, N_BINS, N_CHANNELS)}")
    if arr.dtype != np.float64:
        raise ValueError(f"{path} dtype {arr.dtype}, expected float64")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} contains NaN or inf")
    return arr


def load_raw_aux(path: Path | None = None) -> np.ndarray:
    path = path or array_paths()["x_aux"]
    arr = np.load(path)
    if arr.shape != (N_SAMPLES, N_AUX):
        raise ValueError(f"{path} shape {arr.shape}, expected {(N_SAMPLES, N_AUX)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} contains NaN or inf")
    return arr


def load_importance_index(path: Path | None = None) -> np.ndarray:
    path = path or array_paths()["importance_idx"]
    idx = np.load(path)
    if idx.shape != (N_CNN_FEATURES,):
        raise ValueError(f"{path} shape {idx.shape}, expected {(N_CNN_FEATURES,)}")
    if set(idx.tolist()) != set(range(N_CNN_FEATURES)):
        raise ValueError(f"{path} is not a permutation of 0..{N_CNN_FEATURES - 1}")
    return idx.astype(np.int64, copy=False)


def load_arrays(cfg: dict | None = None):
    """Return X_cnn float32, X_aux float64, and the reference importance permutation."""
    paths = array_paths(cfg)
    x_cnn = load_raw_cnn(paths["x_cnn"]).astype(np.float32, copy=False)
    x_aux = load_raw_aux(paths["x_aux"])
    idx = load_importance_index(paths["importance_idx"])
    return x_cnn, x_aux, idx


def identical_adjacent_pairs(x_aux: np.ndarray) -> np.ndarray:
    """Indices i where row i and row i+1 are exactly equal. Returns the lower index."""
    same = np.all(x_aux[1:] == x_aux[:-1], axis=1)
    return np.flatnonzero(same)


def labels_from_ab_bind(ab_csv: Path) -> pd.DataFrame:
    """Build the canonical label table from Sirin et al. AB_645.csv.

    ``ddg`` is the AB-Bind value. Positive means weaker binding. Non-binders are
    the rows whose value is exactly +8, not -8.
    """
    raw = pd.read_csv(ab_csv)
    if len(raw) != N_SAMPLES:
        raise ValueError(f"{ab_csv} has {len(raw)} rows, expected {N_SAMPLES}")
    rows = []
    for i, rec in raw.iterrows():
        mut = str(rec["Mutation"]).strip()
        match = MUTATION_RE.match(mut)
        if match is None:
            raise ValueError(f"Unparsed mutation at row {i}: {mut}")
        chain, wt, resid, aa = match.group(1), match.group(2), match.group(3), match.group(4)
        pdb = str(rec["#PDB"]).strip()
        ddg = float(rec["ddG(kcal/mol)"])
        rows.append(
            {
                "row_idx": int(i) if not isinstance(i, int) else int(i),
                "pdb": pdb,
                "chain": chain,
                "wt": wt,
                "resid": resid,
                "mut": aa,
                "ddg": ddg,
                "is_nonbinder": int(ddg == NONBINDER_SENTINEL),
                "complex_id": pdb,
                "is_homology": int(pdb.startswith("HM_")),
                "ab_type": str(rec["type"]).strip(),
                "mutation": mut,
            }
        )
    frame = pd.DataFrame(rows)
    frame["row_idx"] = np.arange(len(frame), dtype=int)
    return frame


def _missing_labels_error(path: Path) -> FileNotFoundError:
    cols = ", ".join(REQUIRED_LABEL_COLUMNS)
    return FileNotFoundError(
        f"Missing labels file {path}. "
        f"Expected a CSV with columns: {cols}. "
        "Row order must match X_unnormed.npy / X_aux.npy. "
        "Build it with scripts/build_labels.py from AB_645.csv."
    )


def assert_label_tables(cfg: dict | None = None, supplement: Path | None = None):
    """Startup check for every new script.

    Canonical ``ddg`` is the training target. The supplement is metadata.
    Its ``ddg`` column is the opposite sign and is never a training target.
    Residue identity uses ``resid_str`` (numeric ``resid`` plus ``icode``).
    """
    cfg = cfg or _default_config()
    y, meta = load_labels(cfg=cfg)
    path = Path(supplement) if supplement is not None else repo_supplement_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata table {path}")
    sup = pd.read_csv(
        path,
        dtype={"pdb": str, "chain": str, "wt": str, "mut": str, "resid_str": str, "icode": str},
        keep_default_na=False,
    )
    if len(sup) != len(meta):
        raise AssertionError(f"length {len(meta)} != supplement {len(sup)}")
    if not np.array_equal(sup["row_idx"].to_numpy(), np.arange(len(meta))):
        raise AssertionError(f"{path} row_idx is not 0..{len(meta) - 1} in order")
    for column in ("pdb", "chain", "wt", "mut"):
        left = meta[column].astype(str).str.casefold().to_numpy()
        right = sup[column].astype(str).str.casefold().to_numpy()
        if not np.array_equal(left, right):
            mismatch = int(np.flatnonzero(left != right)[0])
            raise AssertionError(
                f"{column} differs at row_idx {mismatch}: {meta[column].iloc[mismatch]!r} vs {sup[column].iloc[mismatch]!r}"
            )
    resid = meta["resid"].astype(str).str.casefold().to_numpy()
    resid_str = sup["resid_str"].astype(str).str.casefold().to_numpy()
    if not np.array_equal(resid, resid_str):
        mismatch = int(np.flatnonzero(resid != resid_str)[0])
        raise AssertionError(
            f"resid differs from resid_str at row_idx {mismatch}: {meta['resid'].iloc[mismatch]!r} vs {sup['resid_str'].iloc[mismatch]!r}"
        )
    if not np.allclose(y, sup["ddg_abbind"].to_numpy(dtype=np.float64)):
        raise AssertionError("canonical ddg is not allclose to supplement ddg_abbind")
    return y, meta, sup


def repo_supplement_path(cfg: dict | None = None) -> Path:
    cfg = cfg or _default_config()
    return Path(cfg["_root"]) / "data" / "labels" / "ab_bind_s645_supplement.csv"


def load_labels(path: Path | None = None, cfg: dict | None = None):
    """Return y (645,) float64 and the metadata frame. Never reorders rows."""
    cfg = cfg or _default_config()
    path = Path(path) if path is not None else resolve_path(cfg, "labels")
    if not path.is_file():
        raise _missing_labels_error(path)
    meta = pd.read_csv(
        path,
        dtype={
            "pdb": str,
            "chain": str,
            "wt": str,
            "resid": str,
            "mut": str,
            "complex_id": str,
        },
    )
    missing = [c for c in REQUIRED_LABEL_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}. Required: {list(REQUIRED_LABEL_COLUMNS)}"
        )
    if len(meta) != N_SAMPLES:
        raise ValueError(f"{path} has {len(meta)} rows, expected {N_SAMPLES}")
    if meta["ddg"].isna().any():
        raise ValueError(f"{path} contains NaN in ddg")
    row_idx = meta["row_idx"].to_numpy()
    if not np.array_equal(row_idx, np.arange(N_SAMPLES)):
        raise ValueError(
            f"{path} row_idx is not 0..{N_SAMPLES - 1} in order. Refusing to reorder."
        )
    if "is_homology" not in meta.columns:
        raise ValueError(f"{path} is missing is_homology, required for homology_blind splits")
    y = meta["ddg"].to_numpy(dtype=np.float64)
    return y, meta


def aux_alignment_report(x_aux: np.ndarray, meta: pd.DataFrame) -> dict:
    """Compare identical adjacent aux rows with the label table.

    A pair counts as a site match when both rows share complex_id, chain and resid.
    All-zero aux vectors match across different sites; that is reported separately.
    """
    pairs = identical_adjacent_pairs(x_aux)
    site_match = 0
    complex_match = 0
    zero_pairs = 0
    for i in pairs:
        a = meta.iloc[int(i)]
        b = meta.iloc[int(i) + 1]
        if a["complex_id"] == b["complex_id"]:
            complex_match += 1
        if (
            a["complex_id"] == b["complex_id"]
            and a["chain"] == b["chain"]
            and str(a["resid"]) == str(b["resid"])
        ):
            site_match += 1
        if not np.any(x_aux[int(i)]):
            zero_pairs += 1
    n = int(len(pairs))
    return {
        "n_identical_adjacent_pairs": n,
        "same_complex_chain_resid": site_match,
        "same_complex": complex_match,
        "zero_vector_pairs": zero_pairs,
        "site_match_rate": (site_match / n) if n else float("nan"),
    }
