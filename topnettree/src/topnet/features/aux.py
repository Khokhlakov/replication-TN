"""Auxiliary physicochemical features.

The 707-d S645 block is loaded from ``X_aux.npy``. Regenerating it requires the
legacy ``MutationModule`` path (pairwise energies, pKa / MIBPB, environment),
which is not in this package. A substitute vector would not be the training
feature, so this function refuses.
"""


def mutation_aux(*_args, **_kwargs):
    raise NotImplementedError(
        "Auxiliary features are not ported. "
        "Legacy source: gen_feature_aux/MutationModule.py::construct_mutation_feature, "
        "which needs MIBPB, pKa output and environment files. "
        "Refusing to invent a 707-dimensional substitute. Use data/raw/X_aux.npy for S645."
    )
