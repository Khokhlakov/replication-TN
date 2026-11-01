"""Skipped unless TensorFlow/Keras can build the legacy model."""

import numpy as np
import pytest
import torch

tf = pytest.importorskip("tensorflow")

from topnet.cnn import TopNetCNN, keras_flatten


def _legacy():
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input
    from tensorflow.keras.models import Model

    inp = Input(shape=(48, 108))
    h = Conv1D(16, 3, padding="same", activation="relu", kernel_initializer="he_normal")(inp)
    h = Conv1D(16, 3, padding="same", activation="relu", kernel_initializer="lecun_uniform")(h)
    h = Dropout(0.3)(h)
    h = Conv1D(32, 3, padding="same", activation="relu", kernel_initializer="lecun_uniform")(h)
    h = Conv1D(32, 3, padding="same", activation="relu", kernel_initializer="lecun_uniform")(h)
    h = Dropout(0.3)(h)
    flat = Flatten()(h)
    dropped = Dropout(0.1)(flat)
    out = Dense(1, activation="linear")(dropped)
    full = Model(inp, out)
    feat = Model(inp, flat)
    return full, feat


def _copy(keras_model, torch_model):
    convs = [torch_model.c1, torch_model.c2, torch_model.c3, torch_model.c4]
    keras_convs = [layer for layer in keras_model.layers if layer.__class__.__name__ == "Conv1D"]
    for src, dst in zip(keras_convs, convs):
        kernel, bias = src.get_weights()
        dst.weight.data.copy_(torch.from_numpy(np.transpose(kernel, (2, 1, 0)).copy()))
        dst.bias.data.copy_(torch.from_numpy(bias.copy()))
    dense = [layer for layer in keras_model.layers if layer.__class__.__name__ == "Dense"][0]
    kernel, bias = dense.get_weights()
    torch_model.fc.weight.data.copy_(torch.from_numpy(kernel.T.copy()))
    torch_model.fc.bias.data.copy_(torch.from_numpy(bias.copy()))


def test_outputs_and_flatten_match_keras():
    keras_model, feat_model = _legacy()
    torch_model = TopNetCNN()
    _copy(keras_model, torch_model)
    torch_model.eval()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 48, 108)).astype(np.float32)
    k_out = keras_model.predict(x, verbose=0).reshape(-1)
    k_feat = feat_model.predict(x, verbose=0)
    with torch.no_grad():
        xt = torch.from_numpy(x).permute(0, 2, 1)
        t_out = torch_model(xt).numpy()
        t_feat = torch_model.features(xt).numpy()
    assert np.allclose(t_feat, k_feat, atol=1e-5)
    assert np.allclose(t_out, k_out, atol=1e-5)
    assert t_feat.shape[1] == 1536
    _ = keras_flatten
