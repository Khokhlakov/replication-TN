"""Score one mutation from a saved bundle.

``--option 0`` is topo+aux and ``--option 1`` is topo only. Aux is not ported,
so a PDB path can score ``cnn`` and the regenerated H1/H2 sets (``topo``,
``cnn_h12``, ``h12``) only. H0 and H1/H2 for that path come from
``homology_blocks``. A PDB path is refused when the bundle was fit on the
shipped npy arrays, because this generator is not the code that built those
arrays. The output line matches the legacy tool.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from topnet.features.homology import homology_blocks
from topnet.features.manifest import GENERATOR_MANIFEST, manifest_hash
from topnet.pipeline import assert_manifest, load_bundle, predict_pipeline

_NEEDS_AUX = {"topo_aux", "cnn_aux", "aux", "cnn_aux_h12"}
_NEEDS_H12 = {"topo", "h12", "cnn_h12"}

OPTION_SETS = {"0": "topo_aux", "1": "topo"}


def _one_line(value: float) -> str:
    return f"Prediction of binding affinity change is {float(value)} kcal/mol"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--option", choices=["0", "1"], default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--pdb", default=None)
    parser.add_argument("--mut-pdb", default=None)
    parser.add_argument("--chain", default=None)
    parser.add_argument("--wt", default=None)
    parser.add_argument("--resid", default=None)
    parser.add_argument("--mut", default=None)
    parser.add_argument("--cnn-npy", default=None)
    parser.add_argument("--aux-npy", default=None)
    parser.add_argument("--h12-npy", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    bundle = load_bundle(args.bundle)
    feature_set = args.feature_set
    if args.option is not None:
        mapped = OPTION_SETS[args.option]
        if feature_set is None:
            feature_set = mapped
        elif feature_set != mapped:
            print(f"--option {args.option} is {mapped}, not {feature_set}", file=sys.stderr)
            return 2
    if feature_set is None:
        feature_set = bundle["feature_set"]
    if feature_set != bundle["feature_set"]:
        print(
            f"Bundle was fit as {bundle['feature_set']} ({bundle['label']}), not {feature_set}.",
            file=sys.stderr,
        )
        return 2
    if bundle["label"].startswith("no-H12"):
        print(
            f"Scoring ablation '{bundle['feature_set']}' ({bundle['label']}). This is not TopNetTree.",
            file=sys.stderr,
        )

    if args.pdb:
        if bundle["manifest_hash"] != manifest_hash(GENERATOR_MANIFEST):
            print(
                "Refusing PDB inference: the bundle manifest is not generator_v1. "
                "The shipped S645 arrays were not rebuilt by this package, and "
                "auxiliary features are not ported.",
                file=sys.stderr,
            )
            return 2
        if not args.mut_pdb:
            print(
                "A mutant PDB is required. FASPR/PyRosetta are not bundled "
                "(legacy code used SCAP).",
                file=sys.stderr,
            )
            return 2
        if not args.chain or args.resid is None:
            print("--chain and --resid are required with --pdb", file=sys.stderr)
            return 2
        if feature_set in _NEEDS_AUX or feature_set == "h12_original":
            print(
                "PDB scoring can build H0 and H1/H2. Auxiliary features are not ported, "
                f"and '{feature_set}' needs a matrix this path does not have.",
                file=sys.stderr,
            )
            return 2
        try:
            blocks = homology_blocks(
                args.pdb,
                args.mut_pdb,
                args.chain,
                args.resid,
                compute_h12=feature_set in _NEEDS_H12,
            )
        except (ValueError, ImportError) as exc:
            print(exc, file=sys.stderr)
            return 2
        print(
            f"H0 tensor {blocks.h0.shape}"
            + (f", H1/H2 vector {blocks.h12.shape}" if blocks.h12 is not None else ", H1/H2 not requested"),
            file=sys.stderr,
        )
        assert_manifest(bundle, GENERATOR_MANIFEST)
        x_cnn = blocks.h0.reshape(1, 48, 108)
        x_aux = None
        x_h12 = None if blocks.h12 is None else blocks.h12.reshape(1, -1)
    else:
        if not args.cnn_npy and feature_set != "aux":
            print("Provide --cnn-npy or --pdb.", file=sys.stderr)
            return 2
        x_cnn = None if feature_set == "aux" else np.load(args.cnn_npy)
        if x_cnn is not None and x_cnn.ndim == 2:
            x_cnn = x_cnn.reshape(1, 48, 108)
        x_aux = np.load(args.aux_npy) if args.aux_npy else None
        if x_aux is not None and x_aux.ndim == 1:
            x_aux = x_aux.reshape(1, -1)
        x_h12 = np.load(args.h12_npy) if args.h12_npy else None
        if x_h12 is not None and x_h12.ndim == 1:
            x_h12 = x_h12.reshape(1, -1)
        if feature_set in ("topo_aux", "topo") and x_h12 is None:
            print("This option needs --h12-npy. X_h12 was not in the capsule.", file=sys.stderr)
            return 2

    if feature_set == "aux":
        # The CNN still runs inside predict_pipeline. Aux-only bundles were fit
        # with a CNN present only so the artifact stays uniform; the GBDT ignores it.
        if x_cnn is None:
            x_cnn = np.zeros((1, 48, 108), dtype=np.float32)

    pred = predict_pipeline(bundle, x_cnn, x_aux=x_aux, x_h12=x_h12)
    text = _one_line(pred[0])
    if args.out:
        out = Path(args.out)
    elif args.pdb and args.chain and args.wt and args.resid and args.mut:
        stem = Path(args.pdb).stem
        out = Path(f"{stem}_{args.chain}_{args.wt}_{args.resid}_{args.mut}.ddg")
    else:
        out = Path("result.ddg")
    out.write_text(text, encoding="utf-8")
    print(text)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
