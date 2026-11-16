#!/usr/bin/env python3
"""Write the H0 tensor and the H1/H2 vector for one mutation.

The mutant PDB has to already exist. This script does not rebuild a side
chain and it does not fill in a zero vector when the chain or the residue
is missing.

    python scripts/write_homology_blocks.py \
        --wild wild.pdb --mutant mutant.pdb \
        --chain H --resid 100A \
        --out-dir features/H_100A

``--out-dir`` receives ``h0.npy`` with shape (48, 108) and ``h12.npy`` with
shape (756,). ``--h0-only`` skips H1/H2 and does not need Gudhi. If Gudhi
is missing and you did not pass ``--h0-only``, ``h0.npy`` is still written,
``h12.npy`` is not, and the process exits with status 2.

``--legacy-resid`` uses the capsule residue key (PDB columns 24-26). The
default keeps the full residue number and the insertion code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from topnet.features.homology import homology_blocks


def gudhi_available() -> bool:
    try:
        import gudhi  # noqa: F401
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write H0 and H1/H2 blocks for one mutation.")
    parser.add_argument("--wild", required=True, help="Wild-type complex PDB.")
    parser.add_argument("--mutant", required=True, help="Same complex with the mutated side chain.")
    parser.add_argument("--chain", required=True, help="Chain id of the mutated residue.")
    parser.add_argument("--resid", required=True, help="Residue number, optionally with an insertion code (100A).")
    parser.add_argument("--icode", default=None, help="Insertion code, if you did not put it on --resid.")
    parser.add_argument("--cutoff", type=float, default=12.0, help="Neighborhood radius in angstroms. Capsule uses 12.")
    parser.add_argument("--legacy-resid", action="store_true", help="Match PDB columns 24-26, as Gen_pointclouds.py does.")
    parser.add_argument("--no-quantize", action="store_true", help="Keep raw H0 lengths instead of the JavaPlex grid.")
    parser.add_argument("--altloc", choices=["all", "blank", "first"], default="all")
    parser.add_argument("--h0-only", action="store_true", help="Do not compute H1/H2.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    want_h12 = not args.h0_only
    if want_h12 and not gudhi_available():
        try:
            blocks = homology_blocks(
                args.wild,
                args.mutant,
                args.chain,
                args.resid,
                args.icode,
                cutoff=args.cutoff,
                legacy_resid=args.legacy_resid,
                quantize=not args.no_quantize,
                compute_h12=False,
                altloc=args.altloc,
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "h0.npy", blocks.h0)
        print(f"Wrote {out / 'h0.npy'} shape {blocks.h0.shape}")
        print("H1/H2 were not written. Install Gudhi (pip install gudhi) and run again.", file=sys.stderr)
        return 2

    try:
        blocks = homology_blocks(
            args.wild,
            args.mutant,
            args.chain,
            args.resid,
            args.icode,
            cutoff=args.cutoff,
            legacy_resid=args.legacy_resid,
            quantize=not args.no_quantize,
            compute_h12=want_h12,
            altloc=args.altloc,
        )
    except (ValueError, ImportError) as exc:
        print(exc, file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "h0.npy", blocks.h0)
    print(f"Wrote {out / 'h0.npy'} shape {blocks.h0.shape}")
    if blocks.h12 is not None:
        np.save(out / "h12.npy", blocks.h12)
        print(f"Wrote {out / 'h12.npy'} shape {blocks.h12.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
