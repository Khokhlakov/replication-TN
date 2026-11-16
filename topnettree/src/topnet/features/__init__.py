"""Python replacements for the MATLAB / R feature generators.

``homology_blocks`` builds the H0 tensor and the H1/H2 vector from a
wild-type PDB and a mutant PDB.
"""

from topnet.features.homology import homology_blocks
from topnet.features.manifest import GENERATOR_MANIFEST, LEGACY_NPY_MANIFEST, manifest_hash

__all__ = ["GENERATOR_MANIFEST", "LEGACY_NPY_MANIFEST", "homology_blocks", "manifest_hash"]
