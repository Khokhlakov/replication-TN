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
