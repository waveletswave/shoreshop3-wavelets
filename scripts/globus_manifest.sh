#!/usr/bin/env bash
# List the ShoreShop3 Globus collection recursively WITHOUT downloading anything.
#
# Writes one JSON listing per top-level folder to data/manifests/<timestamp>/
# and prints file counts and sizes per team, so you can decide where to keep
# the data (laptop vs. cluster) before running scripts/globus_sync.sh.
#
# Usage:  scripts/globus_manifest.sh          (run `globus login` once first)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/config/globus.env" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/config/globus.env"
fi
SRC="${SHORESHOP3_COLLECTION:-5d24db1e-b934-4e35-9085-513be554aaa7}"
FOLDERS="${SHORESHOP3_FOLDERS:-InputData SubmissionTemplates PublicSubmissions UserSubmissions}"
DEPTH="${MANIFEST_DEPTH:-20}"
PYTHON="${PYTHON:-python3}"

command -v globus >/dev/null 2>&1 || { echo "globus CLI not found: pip install globus-cli" >&2; exit 1; }
globus whoami >/dev/null 2>&1 || { echo "Not logged in to Globus: run 'globus login' first" >&2; exit 1; }

OUT="$ROOT/data/manifests/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
for top in $FOLDERS; do
  echo "Listing /$top/ (up to $DEPTH levels) ..."
  globus ls --all --recursive --recursive-depth-limit "$DEPTH" --format json \
    "$SRC:/$top/" > "$OUT/$top.json"
done

"$PYTHON" "$ROOT/scripts/summarize_manifest.py" "$OUT"
