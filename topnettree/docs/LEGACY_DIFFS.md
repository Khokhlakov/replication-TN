# Differences from the uploaded legacy code

| Legacy behavior | Replacement |
|---|---|
| Python 2 `print`, and `open(..., 'w')` around `np.save` | Python 3. `np.save(path, array)`. |
| `keras.layers.merge.Concatenate`, `Adam(lr=...)`, mixed tabs in `train_model_cnn.py` | PyTorch `TopNetCNN`. Same conv stack, dropout, linear head, Adam `lr=1e-4`, `eps=1e-7`, MSE, batch 8. |
| `predict.py` recomputes mean and std on the training set **plus the new row**, every call, in place | `Normalizer.fit` on training rows only. The stats are stored in the bundle. |
| `normalize` leaves `std == 0` cells unscaled | Default writes 0 there. `legacy_std0=True` keeps the raw value. |
| Pickles loaded from `../../data/gbdt1.pickle` and `cnn_model.h5` | Joblib bundles under `runs/`. Legacy pickles and `h5` files are not loaded. |
| CNN, importance ranking and the feature list are fit once on all 645 rows, then the tree is cross-validated | Default `cv_mode=nested` refits all of those inside each training fold. `leaky_original` keeps the old path and is labeled optimistic. |
| `importance_idx.npy` selects 300 CNN units at inference | That index belongs to the missing `cnn_model.h5`. A new CNN gets a new ranking. The old file is only a bin-profile reference. |
| Final vector is `[cnn_top300, X_h12, X_aux]` or `[cnn_top300, X_h12]` | `X_h12` is absent. Runs use `cnn_aux`, `cnn` or `aux` and are named that way. They are not called TopNetTree. |
| `crossvalidation_ab.py` GBDT: 20000 trees, `lr=0.001`, depth 6, `min_samples_split=3`, `subsample=0.4`, `max_features=sqrt` | `configs/legacy_ab_cv.yaml`. `configs/default.yaml` instead uses the unverified starting point from the task brief (`lr=0.01`, depth 8, `min_samples_split=2`, `subsample=0.7`). Both are marked unverified against the paper SI. |
| Output file is `result.ddg` in the legacy script; the readme uses `1AK4_D_A_488_G.ddg` | The CLI writes the readme filename. The line is still `Prediction of binding affinity change is <value> kcal/mol`. |
| H0 difference channel is wild minus mutant; H12 difference is mutant minus wild | Preserved. See `docs/DATA_NOTES.md`. |
| `gen_feature_12.py` assigns the difference inside the element loop | The final array is the same as assigning it once after the loop. The port assigns it once. |
| Residue id read as `line[23:26]` | Default reader keeps the full resSeq and insertion code. `legacy_resid=True` restores the old slice. |
| MATLAB JavaPlex, R TDA, VMD, SCAP, MIBPB | Not called. H0 uses a bipartite MST. H1/H2 uses Gudhi alpha with `sqrt(filtration)` when Gudhi is installed. Aux generation is refused. |

Flatten order is Keras order: `index = bin * 32 + filter`. The PyTorch activation is permuted before the reshape so this stays true. `tests/test_keras_equivalence.py` checks it against Keras when TensorFlow imports.
