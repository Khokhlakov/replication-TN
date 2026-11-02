import numpy as np
import pytest

from topnet.data import load_labels, load_raw_aux, load_raw_cnn
from topnet.features.manifest import GENERATOR_MANIFEST, manifest_hash
from topnet.pipeline import assert_manifest, fit_pipeline, save_bundle


def test_cli_writes_the_legacy_line_and_refuses_a_mismatched_pdb(cfg, tmp_path, script_loader):
    x = load_raw_cnn()[:16].astype(np.float32)
    aux = load_raw_aux()[:16]
    y, _meta = load_labels()
    y = y[:16]
    bundle = fit_pipeline(x, y, aux, cfg, seed=0, feature_set="cnn_aux", row_idx=np.arange(16))
    path = tmp_path / "bundle.joblib"
    save_bundle(bundle, path)
    cnn_path = tmp_path / "cnn.npy"
    aux_path = tmp_path / "aux.npy"
    np.save(cnn_path, x[:1])
    np.save(aux_path, aux[:1])
    out = tmp_path / "1AK4_D_A_488_G.ddg"
    predict = script_loader("predict")
    code = predict.main(
        [
            "--bundle",
            str(path),
            "--feature-set",
            "cnn_aux",
            "--cnn-npy",
            str(cnn_path),
            "--aux-npy",
            str(aux_path),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("Prediction of binding affinity change is ")
    assert text.endswith(" kcal/mol")
    refused = predict.main(
        ["--bundle", str(path), "--feature-set", "cnn_aux", "--pdb", "1AK4.pdb", "--chain", "D", "--resid", "488"]
    )
    assert refused == 2
    with_option = predict.main(["--bundle", str(path), "--option", "0", "--pdb", "1AK4.pdb"])
    assert with_option == 2
    with pytest.raises(RuntimeError):
        assert_manifest(bundle, GENERATOR_MANIFEST)
    assert bundle["manifest_hash"] != manifest_hash(GENERATOR_MANIFEST)


def test_tune_stays_inside_the_training_rows(cfg, script_loader):
    from topnet.cv import FitLog

    tune = script_loader("tune_gbdt")
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    outer_test = np.arange(10)
    train_idx = np.arange(10, 40)
    log = FitLog()
    best, trials = tune.search_gbdt(features, y, train_idx, cfg, seed=0, fit_log=log)
    assert trials
    assert "rmse" in best
    for event in log.events:
        assert set(event["train_idx"]).isdisjoint(outer_test)
        assert set(event["test_idx"]).isdisjoint(outer_test)
        assert set(event["train_idx"]).issubset(set(train_idx))


def test_random_inner_fit_never_sees_the_outer_test(cfg, monkeypatch, script_loader):
    from sklearn.ensemble import GradientBoostingRegressor

    module = script_loader("tune_gbdt")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 48, 108)).astype(np.float32)
    aux = rng.normal(size=(24, 4))
    y = np.arange(24, dtype=np.float64)
    outer_test = np.arange(6)
    outer_train = np.arange(6, 24)
    seen = []
    real_fit = GradientBoostingRegressor.fit

    def fit_spy(self, x_train, y_train, *args, **kwargs):
        seen.extend(np.asarray(y_train, dtype=np.float64).tolist())
        return real_fit(self, x_train, y_train, *args, **kwargs)

    monkeypatch.setattr(GradientBoostingRegressor, "fit", fit_spy)
    cfg = dict(cfg)
    cfg["tune"] = {
        "inner_splits": 3,
        "n_trials": 1,
        "max_estimators": 4,
        "cnn_option": "A",
        "random_space": {
            "max_depth": [2],
            "subsample": [1.0],
            "learning_rate": [0.1],
            "max_features": ["sqrt"],
            "min_samples_leaf": [1],
            "n_keep": [4],
        },
    }
    cfg["cnn"]["epochs"] = 1
    cfg["select"]["n_estimators"] = 3
    winner, trials = module.random_search_outer(x, y, aux, outer_train, outer_test, cfg, seed=0)
    assert trials
    assert "n_estimators" in winner
    assert set(seen).isdisjoint(set(outer_test.tolist()))
    assert seen
