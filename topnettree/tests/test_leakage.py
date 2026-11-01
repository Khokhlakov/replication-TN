import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from topnet.cnn import train_cnn
from topnet.cv import run_cv
from topnet.data import load_labels, load_raw_aux, load_raw_cnn


def _take(n=48):
    x = load_raw_cnn()[:n].astype(np.float32)
    aux = load_raw_aux()[:n]
    y, _meta = load_labels()
    return x, aux, y[:n]


def test_nested_fit_never_sees_held_out_rows(cfg, monkeypatch):
    x, aux, y = _take()
    seen = []

    def cnn_spy(x_train, y_train, config, seed, device=None):
        seen.append(("cnn", len(x_train)))
        return train_cnn(x_train, y_train, config, seed, device=device)

    real_fit = GradientBoostingRegressor.fit

    def fit_spy(self, x_train, y_train, *args, **kwargs):
        seen.append(("gbdt", len(x_train)))
        return real_fit(self, x_train, y_train, *args, **kwargs)

    monkeypatch.setattr("topnet.cv.train_cnn", cnn_spy)
    monkeypatch.setattr(GradientBoostingRegressor, "fit", fit_spy)
    result = run_cv(x, y, aux, cfg, cv_mode="nested", feature_set="cnn_aux")
    assert result["cv_mode"] == "nested"
    assert "no-H12" in result["label"]
    assert seen
    assert all(n == 32 for _kind, n in seen)
    assert any(kind == "cnn" for kind, _n in seen)
    for event in result["fit_log"].events:
        assert len(np.intersect1d(event["train_idx"], event["test_idx"])) == 0
        assert event["n_train"] == 32
    preds = result["predictions"]
    assert preds["row_idx"].tolist() == list(range(48))
    assert preds["y_pred"].notna().all()


def test_leaky_mode_fits_the_cnn_on_every_row(cfg, monkeypatch):
    x, aux, y = _take()
    seen = []
    real = train_cnn

    def cnn_spy(x_train, y_train, config, seed, device=None):
        seen.append(len(x_train))
        return real(x_train, y_train, config, seed, device=device)

    monkeypatch.setattr("topnet.cv.train_cnn", cnn_spy)
    result = run_cv(x, y, aux, cfg, cv_mode="leaky_original", feature_set="cnn")
    assert seen == [48]
    assert result["label"].startswith("optimistic")
    assert result["headline_ok"] is False
