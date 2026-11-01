"""Hashes that tie an artifact bundle to the feature generator that built its inputs."""

from __future__ import annotations

import hashlib
import json

GENERATOR_MANIFEST = {
    "id": "generator_v1",
    "h0": "bipartite_rips_mst",
    "bins": "48 bins, width 0.25 A, cutoff 12 A",
    "channels": (
        "bind_death, bind_persistence, mutation_death, mutation_persistence; "
        "each block is 9 element pairs x (WT, MUT, WT-MUT)"
    ),
    "h12": "alpha complex, radius = sqrt(gudhi filtration), legacy 7 statistics, DIFF = MUT-WT",
    "aux": "not_ported",
}

LEGACY_NPY_MANIFEST = {
    "id": "legacy_npy_v1",
    "source": "data/raw/X_unnormed.npy + data/raw/X_aux.npy",
    "h12": False,
    "note": "Shipped arrays. Not produced by generator_v1, so pdb inference must refuse.",
}


def manifest_hash(manifest: dict) -> str:
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
