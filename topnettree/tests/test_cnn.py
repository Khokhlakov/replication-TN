import numpy as np
import torch
import torch.nn.functional as F

from topnet.cnn import TopNetCNN, extract_features, keras_flatten, train_cnn
from topnet.data import load_labels, load_raw_cnn
from topnet.normalize import Normalizer


def test_flatten_is_bin_major():
    activation = torch.zeros(1, 32, 48)
    activation[0, 7, 5] = 1
    flat = keras_flatten(activation)
    assert flat.shape == (1, 1536)
    assert flat[0, 5 * 32 + 7] == 1
    assert flat.sum() == 1


def test_forward_shape_and_zero_bias():
    model = TopNetCNN()
    assert torch.count_nonzero(model.c1.bias) == 0
    assert torch.count_nonzero(model.fc.bias) == 0
    fan_in = 108 * 3
    std = (2.0 / fan_in) ** 0.5
    assert float(model.c1.weight.detach().abs().max()) <= 2 * std + 1e-5
    x = torch.randn(4, 108, 48)
    model.eval()
    with torch.no_grad():
        features = model.features(x)
        pred = model(x)
    assert features.shape == (4, 1536)
    assert pred.shape == (4,)
    manual = model.c1(x)
    manual = F.relu(manual)
    manual = F.relu(model.c2(manual))
    manual = model.d1(manual)
    manual = F.relu(model.c3(manual))
    manual = F.relu(model.c4(manual))
    manual = model.d2(manual)
    assert torch.allclose(features, keras_flatten(manual))


def test_overfits_a_64_row_subset():
    raw = load_raw_cnn()[:64]
    y, _meta = load_labels()
    y = y[:64]
    norm = Normalizer().fit(raw)
    x = norm.transform(raw)
    cfg = {
        "device": "cpu",
        "cnn": {
            "in_ch": 108,
            "drop1": 0.3,
            "drop2": 0.3,
            "drop3": 0.1,
            "lr": 1e-4,
            "adam_eps": 1e-7,
            "batch_size": 8,
            "epochs": 900,
            "loss": "mse",
        },
    }
    model = train_cnn(x, y, cfg, seed=0, device="cpu")
    pred = extract_features(model, x)
    # Rebuild the head output in eval mode, which is what a capacity check uses
    # after dropout is switched off. Training itself kept the legacy dropout.
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).permute(0, 2, 1)
        out = model(xt).numpy()
    mse = float(np.mean((out - y) ** 2))
    assert pred.shape == (64, 1536)
    assert mse < 0.05, mse
