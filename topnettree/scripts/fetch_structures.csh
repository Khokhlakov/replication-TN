#!/bin/tcsh -f
# fetch_structures.csh -- Part A wrapper (tcsh). Run from the repo root.
#
#   tcsh scripts/fetch_structures.csh                       # download + validate
#   tcsh scripts/fetch_structures.csh --supplement AB-Bind_supp.xlsx
#   tcsh scripts/fetch_structures.csh --offline             # validate only
#
# Any arguments are passed straight to scripts/fetch_structures.py.

set labels = data/labels/ab_bind_s645.csv
set outdir = data/structures

if (! -e $labels) then
  echo "ERROR: $labels not found (run from the repo root)."
  exit 1
endif

mkdir -p $outdir/raw logs

python scripts/fetch_structures.py --labels $labels --outdir $outdir $argv:q >& logs/fetch_structures.log
set rc = $status
tail -n 60 logs/fetch_structures.log

if ($rc != 0) then
  echo "fetch_structures.py failed (exit $rc). See logs/fetch_structures.log"
  exit $rc
endif

# ---------------------------------------------------------------------------
# Fallback: plain-curl download of anything the python step could not fetch.
# (Only runs when pdb_ids.txt exists and some .pdb/.cif is still missing.)
# ---------------------------------------------------------------------------
foreach id (`cat pdb_ids.txt`)
  if (! -e $outdir/raw/${id}.pdb && ! -e $outdir/raw/${id}.cif) then
    echo "fallback curl: $id"
    curl -fsSL -o $outdir/raw/${id}.pdb https://files.rcsb.org/download/${id}.pdb || \
      curl -fsSL -o $outdir/raw/${id}.cif https://files.rcsb.org/download/${id}.cif || \
      echo "STILL MISSING: $id"
  endif
end

echo ""
echo "Entries in raw/: `ls $outdir/raw | wc -l`"
echo "Re-run with --offline to re-validate after any manual additions:"
echo "  tcsh scripts/fetch_structures.csh --offline"
