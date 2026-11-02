# TopNetTree replication: instructions for the coding agent

Save as `AGENTS.md` (or `.cursor/rules/topnet.mdc`) at the repo root. Read the whole file before writing code.

## 0. Mission

Re-implement the **training and evaluation** of TopNetTree (a CNN on H0 persistence tensors feeding a gradient-boosted-tree regressor) as a **pure Python 3 + PyTorch + scikit-learn** package that runs on a CUDA 12 GPU server. The legacy code is Python 2 / Keras 2.x / scikit-learn 0.18; the feature generators depend on MATLAB (JavaPlex), R (TDA), VMD and legacy binaries, none of which may remain in the final package.

Training set: **AB-Bind S645** (645 single-point mutations, 29 complexes). Target: the binding-free-energy (BFE) change upon mutation, in kcal/mol.

Downstream, the package is used for the Chen & Wei (JPCL 2022) variant analysis. That part ("Layer A") is already validated against `antibodies_BFEs.csv` and must ship as a unit test (section 8).

Work in milestones M0 to M6 (section 6). **Do not start a milestone until the previous one's acceptance tests pass.** Never fabricate data, labels, or "reference" numbers to make a test pass.

---

## 1. Verified facts about the provided files

These were computed from the files, not assumed. Encode them as assertions in `tests/test_data_contract.py`.

| File | Verified content |
|---|---|
| `X_unnormed.npy` | shape `(645, 48, 108)`, float64, integer-valued, min -82, max 1635, no NaN. This is the **CNN input** (H0 tensor): 48 filtration bins × 108 channels. 8.87 % of the (bin, channel) cells have zero std over the 645 samples, and all of them are identically 0. |
| `X_aux.npy` | shape `(645, 707)`, float64, no NaN/inf, 2 constant columns. **Extreme magnitudes**: 48 columns have \|x\| > 1e6, max ≈ 3.7e20 (Lennard-Jones/Coulomb blow-ups). Last column takes integer values -4..7. 36 adjacent row pairs are fully identical. |
| `importance_idx.npy` | shape `(1536,)`, a **permutation of 0..1535** (1536 = 48 × 32, the Keras flatten size). Ordered by descending GBDT importance. `predict.py` uses the first 300. |
| `antibodies.csv` | no header, 186 rows (`PDBID_label`: ACE2 + 185 antibodies). |
| `antibodies_BFEs.csv` | header row, 15 rows × 186 columns (ACE2 + 185 antibodies), no NaN. Rows are the 15 BA.1 RBD mutations, **unlabeled**, in the order G339D, S371L, S373P, S375F, K417N, N440K, G446S, S477N, T478K, E484A, Q493R, G496S, Q498R, N501Y, Y505H. |

Facts about `importance_idx.npy` to use as a sanity check, not as a model component:

- Keras flattens `(48, 32)` channel-last, so `flat_index = bin * 32 + filter`. The first 300 indices fall in bins 10-20 (heavily 12-20), plus smaller clusters around bins 24-26, 31-32 and 42. A re-trained CNN's importance profile **by bin** (`idx // 32`) should look broadly similar. The **filter** index (`idx % 32`) is arbitrary per training run.
- `importance_idx.npy` is tied to the original `cnn_model.h5`, which we do not have. **It cannot be applied to a newly trained CNN.** Use it only for the bin-profile comparison.

Hypotheses about the tensor layout (**verify in M1, do not hard-code**): the 48 bins are probably 0.25 Å steps over 0-12 Å. The 108 channels appear to be blocks of 9 (column means in cols 0-8 and 9-17 are nearly identical, and cols 18-26 have mean ≈ 0, which suggests WT / mutant / difference triplets). Write `scripts/inspect_tensor.py` that prints per-column means grouped in blocks of 9 and reports which blocks look like WT, MUT and DIFF. Record the findings in `docs/DATA_NOTES.md`.

---

## 2. What is NOT available: stop and report, never invent

1. **Labels (`Y`)** for the 645 rows. The legacy code loads `Y_cnn.npy`, which we do not have. They must come from the AB-Bind supplementary table (Sirin et al., Protein Sci. 2016) or from the user.
   - Row order must match `X_unnormed.npy` / `X_aux.npy`. This cannot be inferred from the arrays alone.
   - Expected convention: positive = binding strengthened, the 27 non-binders assigned -8 kcal/mol. Verify the sign on known cases.
   - Create `data/labels/ab_bind_s645.csv` with columns `row_idx, pdb, chain, wt, resid, mut, ddg, is_nonbinder, complex_id`. If this file is missing, the loader must raise a clear error naming the file and columns. `complex_id` is needed for leave-one-complex-out CV (M4).
   - Alignment check to implement: rows whose `X_aux` vectors are identical (36 adjacent pairs) should map to the same (complex, site, residues) in the label table. Report the match rate.
2. **`X_h12`** (the H1/H2 alpha-complex statistics used by the final GBDT). Its dimension is unknown. Until M5 produces it, the final-stage models run on CNN features + aux only, and results are labeled "no-H12".
3. **Trained artifacts** (`cnn_model.h5`, `gbdt1.pickle`, `gbdt2.pickle`). Legacy pickles from scikit-learn 0.18 cannot be loaded in a modern environment. Do not try. Retrain.
4. **Feature-generation code beyond what is uploaded.** `MutationModule.py`, `PrepareStructure.py`, `Run_MIBPB.py`, the MATLAB/R scripts and the binaries (`scap`, `profix`, `mibpb5`, `MS_Intersection`) are absent. Section 7 says how to replace them.
5. A `--synthetic` mode with random labels may exist **only** to smoke-test the code path in CI. Every output it produces must be stamped `SYNTHETIC` in file names and logs.

---

## 3. Audit of the uploaded legacy code

### Quirks to fix (record each in `docs/LEGACY_DIFFS.md`)

| Legacy behavior | Problem | Replacement |
|---|---|---|
| `print "..."`, `open('X.npy','w')` with `np.save` | Python 2 only; text-mode write is wrong | Python 3; `np.save(path, arr)` |
| `from keras.layers.merge import Concatenate`, `Adam(lr=...)` | removed APIs | PyTorch |
| `train_model_cnn.py`: tab/space mixed indentation; the line before `final_output = Dense(1,...)` has a leading space + tab | `IndentationError` in Py3 | rewrite |
| `predict.py::normalize` recomputes mean/std **including the new sample**, normalizes the whole training set again for every prediction, mutates in place | training/inference skew, slow | fit mean/std once on training rows, persist them, apply to new data |
| `normalize` leaves cells with `std == 0` **unscaled** | harmless in training (they are all 0), but a nonzero value at inference leaks through | set `(x - mean) / std` for `std > 0`, and **0** where `std == 0`; keep a `legacy_std0` flag |
| `predict.py` pickles load from `../../data/` | hard-coded paths | config + `joblib` |
| the CNN is trained once on all data, and the importance ranking and GBDT are fitted on features of the same data | CV through this path is leaky | nested CV (section 5.6), with the legacy path available as `cv_mode=leaky_original` for comparison |

### Behavior to preserve exactly

- **Architecture and input orientation.** Keras input is `(48, 108)`: **48 steps, 108 channels**. The convolution slides along the 48 filtration bins.
  `Conv1D(16,3,same,relu,he_normal) → Conv1D(16,3,same,relu,lecun_uniform) → Dropout(0.3) → Conv1D(32,3,same,relu,lecun_uniform) → Conv1D(32,3,same,relu,lecun_uniform) → Dropout(0.3) → Flatten [layers[7], the "CNN feature", 1536-d] → Dropout(0.1) → Dense(1, linear)`.
- Optimizer and loss: Adam, lr 1e-4, MSE, batch 8, 2000 epochs, shuffle each epoch, no validation split, no early stopping.
- CNN feature = the flatten output (**before** the final Dropout).
- Selection: rank the 1536 CNN features by GBDT feature importance, keep the top 300 in rank order.
- Final feature vector = `[cnn_top300, X_h12, X_aux]` (option 0, "topo + aux") or `[cnn_top300, X_h12]` (option 1, "topo only").

---

## 4. Repository layout

```
topnettree/
  pyproject.toml            # python>=3.10, torch (cu124 wheels), scikit-learn, numpy, scipy, pandas, joblib, pyyaml, pytest
  configs/
    default.yaml            # every hyperparameter below, nothing hard-coded in code
    fast.yaml               # epochs=200, n_estimators=2000, for development
  data/
    raw/                    # X_unnormed.npy, X_aux.npy, importance_idx.npy, antibodies*.csv (read-only)
    labels/                 # ab_bind_s645.csv (user-supplied)
  src/topnet/
    data.py                 # loaders + contract checks (section 5.1)
    normalize.py            # Normalizer (section 5.2)
    cnn.py                  # TopNetCNN, train_cnn, extract_features (section 5.3)
    select.py               # importance ranking (section 5.4)
    gbdt.py                 # GBDT wrapper (section 5.5)
    pipeline.py             # fit_pipeline / predict_pipeline, artifact save/load
    cv.py                   # splitters + nested CV runner (section 5.6)
    metrics.py              # section 5.7
    features/               # M5, optional (section 7)
    analysis/layer_a.py     # section 8
  scripts/                  # inspect_tensor.py, run_cv.py, train_final.py, predict.py
  tests/
  docs/                     # DATA_NOTES.md, LEGACY_DIFFS.md, RESULTS.md
```

---

## 5. Specifications

### 5.1 Data loading (`data.py`)

- `load_arrays()` returns `X_cnn (645,48,108) float32`, `X_aux (645,707) float64`, and `idx_ref (1536,)`. Check shapes, dtypes, NaN/inf, and that `idx_ref` is a permutation.
- `load_labels()` returns `y (645,) float64` plus the metadata columns; assert `len == 645` and no NaN.
- Never silently reorder rows. Keep a single canonical `row_idx` through every split and output table.

### 5.2 Normalization (`normalize.py`)

`Normalizer.fit(X_train)`:
- `mean`, `std` computed over axis 0 of the **training rows only**, shape `(48, 108)`.
- `transform(X)` returns `(X - mean) / std` where `std > 0`, else 0. `legacy_std0=True` reproduces the old behavior of leaving the raw value.
- Persist with `joblib`; store a hash of the input array's shape and the training row indices.
- `X_aux` is **not** normalized for the tree models (trees are scale-invariant). Provide `signed_log1p` as an option for any future neural use, and log how many aux values exceed 1e6.

### 5.3 CNN (`cnn.py`, PyTorch)

Tensor orientation: store the data as `(N, 48, 108)`, and feed the model `x.permute(0, 2, 1)` → `(N, 108, 48)` (channels-first for `Conv1d`).

```python
class TopNetCNN(nn.Module):
    def __init__(self, in_ch=108, drop1=0.3, drop2=0.3, drop3=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, 16, 3, padding=1)
        self.c2 = nn.Conv1d(16, 16, 3, padding=1)
        self.c3 = nn.Conv1d(16, 32, 3, padding=1)
        self.c4 = nn.Conv1d(32, 32, 3, padding=1)
        self.d1, self.d2, self.d3 = nn.Dropout(drop1), nn.Dropout(drop2), nn.Dropout(drop3)
        self.fc = nn.Linear(48 * 32, 1)
        # init: c1 he_normal; c2,c3,c4 lecun_uniform; fc glorot_uniform; all biases 0

    def features(self, x):                 # x: (B, 108, 48)
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h)); h = self.d1(h)
        h = F.relu(self.c3(h)); h = F.relu(self.c4(h)); h = self.d2(h)   # (B, 32, 48)
        return h.permute(0, 2, 1).reshape(h.shape[0], -1)               # KERAS ORDER: idx = bin*32 + filter

    def forward(self, x):
        return self.fc(self.d3(self.features(x))).squeeze(-1)
```

**Critical**: `features()` must flatten in Keras order (`bin * 32 + filter`), not PyTorch's default `filter * 48 + bin`. The permutation above does that. `importance_idx.npy` semantics and the by-bin sanity check depend on it.

Initializers (match Keras):
- `he_normal`: truncated normal, std = sqrt(2 / fan_in), `fan_in = in_ch * kernel`.
- `lecun_uniform`: U(±sqrt(3 / fan_in)).
- `Dense`: glorot_uniform.
- Biases: zeros.

Training:
- Adam, `lr=1e-4`, `eps=1e-7` (Keras default; PyTorch defaults to 1e-8), loss `mse`, batch 8, epochs 2000 (configurable), reshuffle every epoch, final partial batch kept.
- The data are tiny (645 × 48 × 108). Put the entire tensor on the GPU and batch by indexing. Avoid `DataLoader` overhead; with 81 steps/epoch × 2000 epochs this matters.
- Seed everything (`torch`, `numpy`, `random`); set `torch.use_deterministic_algorithms(True)` for final runs and log the device and CUDA version.
- `extract_features(model, X)` runs in `eval()` mode under `no_grad`, returns `(N, 1536) float32`.

**Optional equivalence test** (`tests/test_keras_equivalence.py`, skipped when `tensorflow` is not importable): build the legacy Keras model, copy weights into the PyTorch model (Keras conv kernel `(k, in, out)` ↔ torch `(out, in, k)`), and assert that outputs **and** the 1536-d flatten features match to 1e-5. This is the definitive test of the flatten ordering.

### 5.4 Importance ranking (`select.py`)

`rank_cnn_features(F_train, y_train)`: fit `GradientBoostingRegressor` on the 1536 CNN features and return `argsort(-feature_importances_)` (a full permutation). Use the same GBDT hyperparameters as 5.5 unless `select.n_estimators` overrides them. Keep the first `n_keep = 300` (configurable).

Provide `bin_profile(idx, k=300)` → histogram of `idx[:k] // 32` over 48 bins, and a plot comparing it to the profile of the reference `importance_idx.npy`.

### 5.5 GBDT (`gbdt.py`)

Wrapper around `sklearn.ensemble.GradientBoostingRegressor` (faithful) with an optional `xgboost` back-end (`device="cuda"`) for speed. Starting configuration, to be treated as **unverified defaults** (the SI and the plan state no values; the original parameters came from 10 × 10-fold searches):

```yaml
gbdt:
  n_estimators: 20000     # 2000 in fast.yaml
  learning_rate: 0.01
  max_depth: 8
  min_samples_split: 2
  subsample: 0.7
  max_features: sqrt
  loss: squared_error     # sklearn >= 1.0 name for the old 'ls'
  random_state: <seed>
```

Provide `scripts/tune_gbdt.py` with a small search (depth 6-8, subsample 0.4-0.7, n_estimators 3000-20000) using **inner** folds only, never the outer test fold. Log every trial.

Two final models, as in the legacy code:
- **option 0** (`gbdt_topo_aux`): `[cnn_top300 | X_h12 | X_aux]`
- **option 1** (`gbdt_topo`): `[cnn_top300 | X_h12]`

Until `X_h12` exists (M5), run the ablations `[cnn_top300 | X_aux]`, `[cnn_top300]` and `[X_aux]` and name them explicitly. Do not call any of them "TopNetTree".

### 5.6 Cross-validation (`cv.py`)

Splitters:

| Name | Split | Purpose |
|---|---|---|
| `kfold10` | random 10-fold over the 645 rows, repeated with several seeds | paper-style number |
| `loco` | leave-one-complex-out using `complex_id` | honest, out-of-complex number (needs labels metadata) |
| `homology_blind` | train on the 558 non-homology-model mutations, test on the 87 homology-model mutations (needs a flag in the label table) | blind test |

**Nested runner (default, `cv_mode=nested`)**, for each outer fold:
1. Fit `Normalizer` on the training rows.
2. Train a **fresh** CNN on the training rows.
3. Extract 1536-d CNN features for training and test rows.
4. Rank features by GBDT importance **on training rows only**; keep the top 300.
5. Build the final feature matrix and fit the final GBDT on training rows.
6. Predict the held-out rows. Store `row_idx, fold, seed, y_true, y_pred`.

`cv_mode=leaky_original`: train the CNN, the importance ranking and the feature selection once on **all** rows, then run the final GBDT through CV. This mimics the legacy workflow. Report it next to the nested result and label it "optimistic". The gap between the two is a result worth reporting.

Parallelism: one process per fold on a multi-GPU box (`CUDA_VISIBLE_DEVICES`), or sequential on one GPU. The GBDT stage is CPU-bound, so use `joblib` across folds for it.

### 5.7 Metrics (`metrics.py`)

Pearson r, Spearman ρ, RMSE, MAE, computed:
- on all 645 rows,
- **excluding the 27 non-binders**,
- per complex (for `loco`), and as mean/median of per-complex r.

Report mean ± sd across CV repeats. Save predictions as parquet/CSV, metrics as JSON, and the config used.

### 5.8 Reproducibility

Every run writes `runs/<timestamp>_<name>/` containing the config, git hash, package versions, device info, seeds, predictions and metrics. A run must be reproducible from its config alone.

---

## 6. Milestones and acceptance tests

| M | Deliverable | Acceptance |
|---|---|---|
| **M0** | repo scaffold, environment, data contract tests, Layer A test (section 8) | `pytest` passes: shapes/permutation checks from section 1 and the Layer A assertions |
| **M1** | `inspect_tensor.py`; `docs/DATA_NOTES.md` describing the 108-channel layout | findings written down, flagged "inferred" where unverified |
| **M2** | `normalize.py`, `cnn.py`, `select.py`, Keras-equivalence test (if TF available) | forward/flatten match Keras to 1e-5, or documented reason the test was skipped; CNN overfits a 64-sample subset (train MSE < 0.05) |
| **M3** | `gbdt.py`, `pipeline.py`, `cv.py` (nested + leaky), metrics; runs on `fast.yaml` | **requires labels**; end-to-end run with no leakage; the test asserts the held-out rows never appear in any fit call (spy on `fit`) |
| **M4** | full-config 10-fold CV (≥ 3 repeats), `loco`, `homology_blind`; `docs/RESULTS.md` | targets below; failures reported, not hidden |
| **M5** | Python H1/H2 + aux feature generation (section 7) | see section 7 |
| **M6** | inference CLI + variant-panel analysis (section 8) | G0 passes, CLI reproduces the CV-held-out predictions |

**Reference targets for M4** (taken from the replication plan's addendum, which cites the TopNetTree paper; verify them against Wang, Cang & Wei, Nat. Mach. Intell. 2020 before declaring pass or fail):

- S645 10-fold CV: Pearson r ≈ 0.65, RMSE ≈ 1.57 kcal/mol; without the 27 non-binders r ≈ 0.68, RMSE ≈ 1.06.
- Blind test on the homology-model subset: r ≈ 0.55.
- Leave-one-complex-out: mean r ≈ 0.51.

Acceptance: nested `kfold10` r within ±0.05 of 0.65, **or** a documented explanation (for example, the absence of `X_h12` in the feature set). Report `leaky_original` separately and never present it as the headline number.

---

## 7. M5: Python feature generation (only after M4)

Goal: remove MATLAB/R/Java and produce `X_cnn`-style H0 tensors, `X_h12`, and a modern replacement for the aux block, so new mutations can be scored. Follow `Replication Blueprint` Phase 2-3 and Addendum A.4 for details. Key rules:

- **H0 (Rips)**: single-linkage on the bipartite distance (infinite within the same set). The finite death values are the MST edge lengths: `scipy.sparse.csgraph.minimum_spanning_tree`. Cross-check against `gudhi.RipsComplex` on random clouds (1e-9).
- **H1/H2 (alpha)**: `gudhi.AlphaComplex`; use `sqrt(filtration)` (radius scale, **no ×2**) per the paper's definition. Statistics only (sum, max, min, mean, std of bar lengths, births, deaths), no binning.
- **Tensor spec**: 48 bins over 0-12 Å (confirm from M1), per-bin counts of intervals, births, deaths. Match the channel layout found in M1.
- **Reference check**: reproduce `X_unnormed.npy` rows for a few AB-Bind mutations from structures. Exact match is the goal; if not exact, report which channel blocks differ.
- Mutant structures: FASPR or PyRosetta in place of SCAP/Jackal; PDB2PQR/PROPKA/APBS in place of MIBPB; FreeSASA/EDTSurf in place of ESES. Label every substituted block (`sasa_proxy`, `ss_substituted`, ...) in the feature manifest.
- Unit tests: rigid-motion and permutation invariance, no-op mutation gives all-zero DIFF, determinism (same job twice gives identical vectors).

---

## 8. M6: inference and the variant-panel analysis

### Inference CLI

`scripts/predict.py --pdb 1AK4.pdb --chain D --wt A --resid 488 --mut G --option 0|1` → writes `1AK4_D_A_488_G.ddg` with the same one-line format as the legacy tool. It loads the persisted Normalizer, CNN, importance indices and GBDT from one artifact bundle and refuses to run if the feature manifest hash differs from the bundle's.

### Layer A (exact, already verified; implement as `analysis/layer_a.py` and `tests/test_layer_a.py`)

```python
THRESH = [0, -0.3, -0.6, -0.9, -1.2, -1.5, -1.8, -2.1, -2.4, -2.7, -3.0]

def disrupted_counts(P, thresholds=THRESH):
    """P: (n_mutations, n_complexes). A complex is disrupted at t iff its most
    negative single-mutation BFE change is strictly < t."""
    m = np.nanmin(P, axis=0)
    return [int((m < t).sum()) for t in thresholds]

def fold_reduction(P, kT=1.0):
    """exp(-mean over complexes of the sum over mutations of min(P, 0)). kT = 1 is the paper convention."""
    return float(np.exp(-np.clip(P, None, 0.0).sum(axis=0).mean() / kT))
```

```python
B = pd.read_csv("antibodies_BFEs.csv"); P = B.drop(columns=["ACE2"]).to_numpy()
assert disrupted_counts(P) == [185, 163, 143, 124, 106, 78, 60, 33, 15, 5, 2]
assert abs(fold_reduction(P) - 20.94) < 0.05          # computed: 20.9445
assert abs(B["ACE2"].sum() - 2.602) < 1e-3            # computed: 2.6018
```

Details that matter: the comparison is strict (`<`, not `<=`); use the exact threshold list, never float `arange`; the exponent has **no RT** (`kT=0.593` is a labeled sensitivity option only); infectivity ratio is `exp(sum_a - sum_b)`.

### Applying the S645-trained model to the panel

Needs M5. Compute 26 unique RBD mutations × 186 complexes (variant → mutation sets are in the replication blueprint, section 1), then compare with `antibodies_BFEs.csv` (BA.1 rows).

**Expectation**: `antibodies_BFEs.csv` was produced by TopNetmAb (trained on SKEMPI 2.0 + deep-mutational-scanning data), **not** an AB-Bind-trained TopNetTree. Report Pearson/Spearman, sign agreement where |published| > 0.3, the disrupted-set Jaccard at -0.3, and the Layer A statistics of *your* predictions. These are exploratory comparisons, **not gates**. A weak match reflects the different training data and is not by itself a bug. Sign check: K417N × LY-CoV016 should be strongly negative (published ≈ -2.0) and N501Y × ACE2 positive.

---

## 9. Environment (CUDA 12)

- Python ≥ 3.10; `torch` from the cu124 wheel index (`pip install torch --index-url https://download.pytorch.org/whl/cu124`).
- scikit-learn ≥ 1.3, numpy, scipy, pandas, joblib, pyarrow, pyyaml, pytest. Optional: `xgboost` (CUDA), `gudhi`, `tensorflow-cpu` (only for the Keras-equivalence test), `freesasa`, `biopython`.
- Provide a `Makefile` (`make test`, `make cv-fast`, `make cv-full`) and a pinned `requirements.lock` after the first successful build.
- No code path may call MATLAB, R, Java, VMD or external binaries in M0-M4.

---

## 10. Rules for the agent

1. **No data fabrication.** If labels or any required file are missing, raise an error that names the file and schema, then continue with the parts that do not need it (M0-M2).
2. **No train/test contamination.** Normalization, CNN training, importance ranking, feature selection and GBDT fitting all use training rows only in `nested` mode. Add the `fit`-spy test.
3. Preserve row identity (`row_idx`) everywhere; never shuffle arrays without carrying it along.
4. Every deviation from the legacy code goes into `docs/LEGACY_DIFFS.md`; every guessed hyperparameter is tagged `# UNVERIFIED` in the config.
5. Do not unpickle legacy `.pickle` / `.h5` artifacts and do not try to recover weights.
6. Keep configs declarative (YAML); no magic numbers in code.
7. Small, reviewable commits per milestone; run `pytest` before each commit.
8. When uncertain about a scientific choice (bin width, channel meaning, label sign), write the options and your evidence into `docs/DATA_NOTES.md` and ask; do not silently pick.

---

## 11. Final report template (`docs/RESULTS.md`)

For each reference number: published value, your value, absolute and relative error, pass/fail against the tolerance fixed above, the milestone it belongs to. Include: nested vs leaky CV gap, with/without non-binders, the no-H12 ablations, the by-bin importance profile vs the reference, runtime per fold, and a list of every substitution made relative to the original pipeline.
