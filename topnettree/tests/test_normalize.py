import numpy as np

from topnet.normalize import Normalizer, count_large, signed_log1p


def test_fit_ignores_future_rows_and_zero_std():
    train = np.zeros((4, 2, 2), dtype=np.float64)
    train[:, 0, 0] = [1, 2, 3, 4]
    train[:, 0, 1] = 5
    norm = Normalizer().fit(train, row_idx=np.array([10, 11, 12, 13]))
    assert norm.mean_[0, 0] == 2.5
    new = np.array([[[9.0, 7.0], [0.0, 0.0]]])
    scaled = norm.transform(new)
    std = train[:, 0, 0].std()
    assert np.isclose(scaled[0, 0, 0], (9.0 - 2.5) / std)
    assert scaled[0, 0, 1] == 0
    legacy = norm.transform(new, legacy_std0=True)
    assert legacy[0, 0, 1] == 7
    assert list(norm.train_rows_) == [10, 11, 12, 13]


def test_signed_log_and_large_count():
    values = np.array([-np.e + 1, 0.0, 1e7])
    logged = signed_log1p(values)
    assert np.isclose(logged[0], -1.0)
    assert logged[1] == 0
    assert count_large(values) == 1


def test_joblib_roundtrip(tmp_path):
    train = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    norm = Normalizer().fit(train)
    path = tmp_path / "norm.joblib"
    norm.save(path)
    loaded = Normalizer.load(path)
    assert np.allclose(loaded.transform(train), norm.transform(train))
