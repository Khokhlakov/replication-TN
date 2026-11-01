import numpy as np

from topnet.config import repo_root
from topnet.data import (
    N_AUX,
    N_CNN_FEATURES,
    N_SAMPLES,
    aux_alignment_report,
    identical_adjacent_pairs,
    load_arrays,
    load_importance_index,
    load_labels,
    load_raw_aux,
    load_raw_cnn,
)
from topnet.select import bin_profile


def test_cnn_array_contract():
    raw = load_raw_cnn()
    assert raw.shape == (N_SAMPLES, 48, 108)
    assert raw.dtype == np.float64
    assert np.isfinite(raw).all()
    assert raw.min() == -82
    assert raw.max() == 1635
    assert np.all(raw == np.round(raw))
    std = raw.std(axis=0)
    zero = std == 0
    assert abs(float(zero.mean()) - 0.08873456790123457) < 1e-12
    assert np.all(raw[:, zero] == 0)
    loaded, _, _ = load_arrays()
    assert loaded.dtype == np.float32
    assert loaded.shape == raw.shape


def test_aux_array_contract():
    aux = load_raw_aux()
    assert aux.shape == (N_SAMPLES, N_AUX)
    assert aux.dtype == np.float64
    assert np.isfinite(aux).all()
    std = aux.std(axis=0)
    assert int((std == 0).sum()) == 2
    big = np.max(np.abs(aux), axis=0) > 1e6
    assert int(big.sum()) == 48
    assert aux.max() > 1e20
    last = set(np.unique(aux[:, -1]).tolist())
    assert last == set(range(-4, 8))
    assert len(identical_adjacent_pairs(aux)) == 36


def test_importance_permutation_and_bins():
    idx = load_importance_index()
    assert idx.shape == (N_CNN_FEATURES,)
    assert set(idx.tolist()) == set(range(N_CNN_FEATURES))
    profile = bin_profile(idx, 300)
    assert int(profile[12:21].sum()) > 150
    assert int(profile.argmax()) in range(12, 21)


def test_labels_and_aux_alignment():
    y, meta = load_labels()
    assert y.shape == (N_SAMPLES,)
    assert not np.isnan(y).any()
    assert int(meta["is_nonbinder"].sum()) == 27
    assert int(meta["is_homology"].sum()) == 87
    assert meta["complex_id"].nunique() == 29
    assert set(meta["resid"].astype(str)) >= {"100a", "488"}
    report = aux_alignment_report(load_raw_aux(), meta)
    assert report["n_identical_adjacent_pairs"] == 36
    assert report["same_complex_chain_resid"] == 3
    assert report["same_complex"] == 36
    assert report["zero_vector_pairs"] > 3
    y_path = repo_root().parent / "capsule-2202829" / "code" / "crossvalidation" / "Y.npy"
    if y_path.is_file():
        raw = np.load(y_path, allow_pickle=True)
        decoded = np.array([float(v.decode() if isinstance(v, (bytes, bytearray)) else v) for v in raw])
        assert np.allclose(decoded, y)


def test_missing_labels_names_the_schema(tmp_path):
    from topnet.data import load_labels as load

    try:
        load(tmp_path / "ab_bind_s645.csv")
    except FileNotFoundError as exc:
        text = str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
    for column in ("row_idx", "pdb", "chain", "wt", "resid", "mut", "ddg", "is_nonbinder", "complex_id"):
        assert column in text
