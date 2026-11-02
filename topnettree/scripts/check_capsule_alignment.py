"""G0 row-order and label gate. Read-only. Writes docs/ALIGNMENT_G0.md.

Canonical training labels stay data/labels/ab_bind_s645.csv. The supplement is
metadata. Nothing here reorders or overwrites those files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from topnet.config import load_config, repo_root
from topnet.data import assert_label_tables

CAPSULE = Path(__file__).resolve().parents[2] / "capsule-2202829"


def _decode_y(raw: np.ndarray) -> np.ndarray:
    values = []
    for item in np.asarray(raw).reshape(-1):
        if isinstance(item, bytes):
            item = item.decode()
        values.append(float(item))
    return np.asarray(values, dtype=np.float64)


def _file_bytes_equal(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()


def _constant_columns(matrix: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.std(matrix, axis=0) == 0)


def _standardize(matrix: np.ndarray):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    constant = std == 0
    out = np.zeros(matrix.shape, dtype=np.float64)
    keep = ~constant
    out[:, keep] = (matrix[:, keep] - mean[keep]) / std[keep]
    return out, constant


def _best_correlation(x_std: np.ndarray, source_std: np.ndarray, batch: int = 256):
    """Each X column's best |Pearson| and the source column that attains it."""
    n = x_std.shape[0]
    n_x = x_std.shape[1]
    best = np.zeros(n_x, dtype=np.float64)
    which = np.zeros(n_x, dtype=np.int64)
    for start in range(0, n_x, batch):
        block = x_std[:, start : start + batch]
        corr = (block.T @ source_std) / n
        mag = np.abs(corr)
        which[start : start + block.shape[1]] = np.argmax(mag, axis=1)
        best[start : start + block.shape[1]] = mag[np.arange(block.shape[1]), np.argmax(mag, axis=1)]
    return best, which


def _counts(values: np.ndarray) -> str:
    return (
        f"|r|>0.99999: {int((values > 0.99999).sum())}; "
        f">0.999: {int((values > 0.999).sum())}; "
        f">0.99: {int((values > 0.99).sum())}; "
        f"<0.9: {int((values < 0.9).sum())}"
    )


def _order_note(indices: np.ndarray) -> str:
    if len(indices) == 0:
        return "no columns"
    steps = np.diff(indices.astype(np.int64))
    if np.all(steps == 1):
        return f"original order, contiguous from {int(indices[0])}"
    if np.all(steps > 0):
        return "increasing but not contiguous"
    return "permuted"


def _match_ab645(supplement: pd.DataFrame) -> str:
    path = CAPSULE / "data" / "AB-Bind-Database" / "AB_645.csv"
    raw = pd.read_csv(path)
    lines = []
    if len(raw) != len(supplement):
        lines.append(f"AB_645.csv has {len(raw)} rows, supplement has {len(supplement)}.")
    mutation = (
        supplement["chain"].astype(str)
        + ":"
        + supplement["wt"].astype(str)
        + supplement["resid_str"].astype(str)
        + supplement["mut"].astype(str)
    )
    def _key(pdb, mut, value):
        return (str(pdb).casefold(), str(mut).casefold(), float(np.round(value, 6)))

    left = [_key(p, m, v) for p, m, v in zip(raw["#PDB"], raw["Mutation"], raw["ddG(kcal/mol)"].to_numpy(dtype=np.float64))]
    right = [
        _key(p, m, v)
        for p, m, v in zip(supplement["pdb"], mutation, supplement["ddg_abbind"].to_numpy(dtype=np.float64))
    ]
    same = left == right
    lines.append(f"Row-by-row (pdb, mutation, rounded ddg) equality: {same}.")
    if same:
        lines.append("Permutation is the identity. Insertion codes are inside the mutation string.")
        return "\n".join(lines)
    from collections import defaultdict

    buckets = defaultdict(list)
    for index, key in enumerate(right):
        buckets[key].append(index)
    permutation = []
    unresolved = []
    for index, key in enumerate(left):
        choices = buckets.get(key) or []
        if not choices:
            unresolved.append(index)
            permutation.append(-1)
        else:
            permutation.append(choices.pop(0))
    lines.append(f"Greedy match on (pdb, mutation, value) unresolved AB_645 rows: {len(unresolved)}.")
    if unresolved:
        lines.append(f"First unresolved AB_645 index: {unresolved[0]}.")
    else:
        permutation = np.asarray(permutation)
        lines.append(f"Matched permutation equals identity: {bool(np.array_equal(permutation, np.arange(len(permutation))))}.")
        moved = np.flatnonzero(permutation != np.arange(len(permutation)))
        lines.append(f"Rows whose position differs: {len(moved)}.")
        if len(moved):
            show = moved[:12]
            lines.append(
                "First moved pairs (AB_645 index -> supplement row_idx): "
                + ", ".join(f"{int(i)}->{int(permutation[i])}" for i in show)
            )
    return "\n".join(lines)


def _crosslabel_report(path: Path) -> str:
    values = np.load(path)
    flat = np.asarray(values).reshape(-1)
    unique, counts = np.unique(flat, return_counts=True)
    repeated = unique[counts > 1]
    missing = sorted(set(range(int(flat.max()) + 1)) - set(int(v) for v in unique))
    lines = [
        f"shape {values.shape} dtype {values.dtype}",
        f"{len(unique)} unique values, min {float(flat.min()):g}, max {float(flat.max()):g}.",
        "Values that occur more than once: " + ", ".join(f"{v:g} x{int(c)}" for v, c in zip(unique, counts) if c > 1),
        f"Integers missing between 0 and the max: {missing[:20]}",
        f"Equals 0..644: {bool(np.array_equal(flat, np.arange(len(flat))))}.",
    ]
    lines.append(
        "crossvalidation_ab.py lines 14-15 load crosslabel.npy and cast it to int, and both lines are commented out. "
        "The script that actually runs does not read this array. It is not a fold id used by the CV loop."
    )
    if len(unique) <= 3 and set(np.round(unique, 6)).issubset({0.0, 1.0}):
        lines.append("The values look like a binary class label. What the class means is not stated in the CV script.")
    elif len(unique) in (5, 10) and np.all(unique == np.arange(len(unique))):
        lines.append("The values look like fold ids, but the running script does not use them.")
    else:
        lines.append(
            "The value pattern is not a 10-fold id and not an obvious two-class label. Meaning: unknown."
        )
    return "\n".join(lines)


def _column_report(x: np.ndarray, aux: np.ndarray, cnn: np.ndarray) -> str:
    cnn_flat = cnn.reshape(cnn.shape[0], -1)
    aux_const = _constant_columns(aux)
    cnn_const = _constant_columns(cnn_flat)
    x_const = _constant_columns(x)
    lines = [
        f"X.npy shape {x.shape}. Constant columns: {len(x_const)}. "
        f"Column std min {float(x.std(axis=0).min()):.4f}, max {float(x.std(axis=0).max()):.4f} "
        "(a z-scored column would have std 1).",
        f"X_aux constant columns ({len(aux_const)}): {aux_const.tolist()}.",
        f"X_unnormed flattened constant columns: {len(cnn_const)} of {cnn_flat.shape[1]}.",
        "Exact column identity was already 0. This section is correlation after dropping constant source columns.",
    ]
    aux_keep = np.delete(aux, aux_const, axis=1)
    cnn_keep = np.delete(cnn_flat, cnn_const, axis=1)
    aux_std, _ = _standardize(aux_keep)
    cnn_std, _ = _standardize(cnn_keep)
    x_std, _ = _standardize(x)
    aux_r, aux_which = _best_correlation(x_std, aux_std)
    cnn_r, cnn_which = _best_correlation(x_std, cnn_std)
    use_aux = aux_r >= cnn_r
    best = np.where(use_aux, aux_r, cnn_r)
    lines.append(
        f"Best |Pearson r| summary: max {float(best.max()):.4f}, median {float(np.median(best)):.4f}, "
        f"95th percentile {float(np.percentile(best, 95)):.4f}."
    )
    lines.append("Best |Pearson r| of each X.npy column against the remaining sources: " + _counts(best))
    x_rank = stats.rankdata(x, axis=0)
    aux_rank = stats.rankdata(aux_keep, axis=0)
    cnn_rank = stats.rankdata(cnn_keep, axis=0)
    x_rank_std, _ = _standardize(x_rank)
    aux_rank_std, _ = _standardize(aux_rank)
    cnn_rank_std, _ = _standardize(cnn_rank)
    aux_s, _ = _best_correlation(x_rank_std, aux_rank_std)
    cnn_s, _ = _best_correlation(x_rank_std, cnn_rank_std)
    best_s = np.maximum(aux_s, cnn_s)
    lines.append(
        f"Best |Spearman rho| summary: max {float(best_s.max()):.4f}, median {float(np.median(best_s)):.4f}."
    )
    lines.append("Best |Spearman rho|: " + _counts(best_s))
    monotone = int(((best_s > 0.999) & (best <= 0.99)).sum())
    lines.append(
        f"Columns with |Spearman|>0.999 and |Pearson|<=0.99 (monotone but not linear): {monotone}."
    )
    aux_idx = np.flatnonzero(use_aux)
    cnn_idx = np.flatnonzero(~use_aux)
    lines.append(
        f"Assigned to aux: {len(aux_idx)}. Assigned to flattened X_unnormed: {len(cnn_idx)}."
    )
    weak = np.flatnonzero(best < 0.9)
    if len(weak) == len(best):
        lines.append(
            "Every X.npy column has best |r| < 0.9 against both sources. "
            "That rejects a column-wise affine copy of X_aux or X_unnormed, including a z-scored concatenation. "
            "It is not a located H1/H2 block: no source block matched, so there is no remainder to cut out. T4.8 is skipped."
        )
    elif len(weak):
        breaks = np.where(np.diff(weak) > 1)[0]
        ranges = []
        start = weak[0]
        for cut in breaks:
            stop = weak[cut]
            ranges.append((int(start), int(stop), int(stop - start + 1)))
            start = weak[cut + 1]
        ranges.append((int(start), int(weak[-1]), int(weak[-1] - start + 1)))
        lines.append(
            "X.npy columns with best |r| < 0.9 (candidate extra block): "
            + ", ".join(f"{a}:{b} (n={n})" for a, b, n in ranges)
        )
    else:
        lines.append("No X.npy column has best |r| < 0.9. No unmatched extra block.")
    aux_sources = aux_which[aux_idx]
    cnn_sources = cnn_which[cnn_idx]
    lines.append(
        f"Aux assignment unique: {len(np.unique(aux_sources)) == len(aux_sources)} "
        f"({len(np.unique(aux_sources))} distinct / {len(aux_sources)} columns, source width {aux_keep.shape[1]})."
    )
    lines.append(
        f"CNN assignment unique: {len(np.unique(cnn_sources)) == len(cnn_sources)} "
        f"({len(np.unique(cnn_sources))} distinct / {len(cnn_sources)} columns, source width {cnn_keep.shape[1]})."
    )
    lines.append("Aux-assigned X columns, in X order: " + _order_note(aux_sources) + ".")
    lines.append("CNN-assigned X columns, in X order: " + _order_note(cnn_sources) + ".")
    if len(cnn_idx):
        lines.append(
            f"CNN-assigned X index range {int(cnn_idx[0])}:{int(cnn_idx[-1])}. "
            f"Aux-assigned X index range {int(aux_idx[0]) if len(aux_idx) else 'none'}:"
            f"{int(aux_idx[-1]) if len(aux_idx) else 'none'}."
        )
    importance = CAPSULE / "code" / "crossvalidation" / "importanceidx.npy"
    if importance.is_file():
        idx = np.load(importance)
        lines.append(
            f"importanceidx.npy shape {idx.shape}. crossvalidation_ab.py:16-20 applies idx[:300] to all 645 rows before KFold. "
            "That selection sees every row. The script that computed the ranking is not in the capsule, "
            "so 'fit on all 645 rows' is the implication of one global index, not a logged training call. "
            "Using it for feature selection is a leak either way."
        )
    high = best > 0.99999
    lines.append(
        f"Columns with |r|>0.99999: {int(high.sum())} / {len(best)}. "
        "A value this close to 1 is an affine match (z-scoring), not a second feature family."
    )
    return "\n".join(lines)


def main() -> int:
    cfg = load_config("configs/default.yaml")
    y, _meta, supplement = assert_label_tables(cfg)
    root = repo_root()
    y_path = CAPSULE / "code" / "crossvalidation" / "Y.npy"
    y_capsule = _decode_y(np.load(y_path, allow_pickle=True))
    y_match = int(np.sum(np.isclose(y_capsule, supplement["ddg_abbind"].to_numpy(dtype=np.float64))))
    if y_match == len(y) and np.allclose(y_capsule, y):
        verdict = "PASS"
        verdict_text = (
            f"Y.npy equals supplement ddg_abbind and canonical ddg at {y_match}/{len(y)} positions, in order. "
            "G0 PASS. T1-T5 may proceed. This re-runs the check already done; it does not re-derive the labels."
        )
    elif set(np.round(y_capsule, 6)) == set(np.round(y, 6)):
        verdict = "FAIL"
        verdict_text = (
            "Y.npy is a permutation of ddg_abbind, not the same order. "
            "STOP T1-T4. The baseline r=0.467 is INVALID until the nested cnn_aux run is repeated on the reordered table."
        )
    else:
        verdict = "STOP"
        verdict_text = "Y.npy values differ from ddg_abbind. STOP. No reordered table was written."

    aux_repo = root / "data" / "raw" / "X_aux.npy"
    cnn_repo = root / "data" / "raw" / "X_unnormed.npy"
    aux_cap = CAPSULE / "code" / "src" / "model" / "X_aux.npy"
    cnn_cap = CAPSULE / "code" / "src" / "model" / "X_unnormed.npy"
    x = np.load(CAPSULE / "code" / "crossvalidation" / "X.npy")
    aux = np.load(aux_repo)
    cnn = np.load(cnn_repo)

    parts = [
        "# G0 alignment",
        "",
        f"Verdict: **{verdict}**",
        "",
        verdict_text,
        "",
        "## a. Y.npy versus ddg_abbind",
        "",
        f"Equal positions: {y_match} / {len(y)}. allclose to canonical ddg: {bool(np.allclose(y_capsule, y))}.",
        "Provenance: capsule `code/crossvalidation/Y.npy`, decoded from byte strings and cast to float.",
        "",
        "## b. crosslabel.npy",
        "",
        _crosslabel_report(CAPSULE / "code" / "crossvalidation" / "crosslabel.npy"),
        "",
        "## c. X.npy columns",
        "",
        _column_report(x, aux, cnn),
        "",
        "If the aux block of X.npy correlates at about 1 with X_aux, the capsule's X.npy/Y.npy pairing independently verifies the X_aux row order.",
        "",
        "## d. AB_645.csv versus the supplement",
        "",
        _match_ab645(supplement),
        "",
        "## e. Repo arrays versus capsule model arrays",
        "",
        f"X_aux.npy byte-identical: {_file_bytes_equal(aux_repo, aux_cap)}.",
        f"X_unnormed.npy byte-identical: {_file_bytes_equal(cnn_repo, cnn_cap)}.",
        "",
        "## Decision",
        "",
        verdict_text,
        "",
    ]
    text = "\n".join(parts)
    out = root / "docs" / "ALIGNMENT_G0.md"
    out.write_text(text, encoding="utf-8")
    print(verdict)
    print(out)
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
