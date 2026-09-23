#!/usr/bin/env bash
# One-way mirror: ShoreShop3 Globus collection  ->  data/raw/ (or another collection).
#
# * Only READS from the ShoreShop3 collection; nothing is ever written there.
# * Re-running transfers only new or changed files (checksum comparison), so it
#   is safe to run again whenever teams resubmit.
# * Nothing is deleted on the destination either.
#
# Usage:
#   scripts/globus_sync.sh                    all folders in SHORESHOP3_FOLDERS
#   scripts/globus_sync.sh SubmissionTemplates InputData      only these
#   DRY_RUN=1 scripts/globus_sync.sh          print the transfer request only
#   WAIT=0 scripts/globus_sync.sh             submit and return immediately
#
# Needs: `globus login`, and Globus Connect Personal running on this computer
# (or DEST_COLLECTION / DEST_ROOT set in config/globus.env).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/config/globus.env" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/config/globus.env"
fi
SRC="${SHORESHOP3_COLLECTION:-5d24db1e-b934-4e35-9085-513be554aaa7}"
if [ "$#" -gt 0 ]; then
  FOLDERS="$*"
else
  FOLDERS="${SHORESHOP3_FOLDERS:-InputData SubmissionTemplates PublicSubmissions UserSubmissions}"
fi

command -v globus >/dev/null 2>&1 || { echo "globus CLI not found: pip install globus-cli" >&2; exit 1; }
globus whoami >/dev/null 2>&1 || { echo "Not logged in to Globus: run 'globus login' first" >&2; exit 1; }

if [ -n "${DEST_COLLECTION:-}" ]; then
  DEST="$DEST_COLLECTION"
else
  DEST="$(globus endpoint local-id 2>/dev/null || true)"
fi
if [ -z "$DEST" ]; then
  echo "No destination found. Install and start Globus Connect Personal," >&2
  echo "or set DEST_COLLECTION in config/globus.env." >&2
  exit 1
fi
if [ "$DEST" = "$SRC" ]; then
  echo "Refusing to transfer the ShoreShop3 collection onto itself." >&2
  exit 1
fi

if [ -z "${DEST_ROOT:-}" ]; then
  if [ -n "${DEST_COLLECTION:-}" ]; then
    echo "Set DEST_ROOT in config/globus.env for DEST_COLLECTION." >&2
    exit 1
  fi
  # Globus Connect Personal addresses your home folder as a literal "~"
  # shellcheck disable=SC2088
  case "$ROOT" in
    "$HOME"/*) DEST_ROOT="~/${ROOT#"$HOME"/}/data/raw" ;;
    *) echo "This repo is outside your home folder: set DEST_ROOT in config/globus.env." >&2
       exit 1 ;;
  esac
fi

BATCH="$(mktemp)"
trap 'rm -f "$BATCH"' EXIT
for top in $FOLDERS; do
  printf -- '--recursive "/%s/" "%s/%s/"\n' "$top" "$DEST_ROOT" "$top" >> "$BATCH"
done

echo "From: $SRC (read only)"
echo "To:   $DEST : $DEST_ROOT"
sed 's/^/  /' "$BATCH"

ARGS=(transfer "$SRC" "$DEST" --batch "$BATCH" --sync-level checksum
      --label "ShoreShop3 mirror $(date +%Y-%m-%d)")
if [ "${DRY_RUN:-0}" = "1" ]; then
  globus "${ARGS[@]}" --dry-run
  exit 0
fi

TASK_ID="$(globus "${ARGS[@]}" --jmespath task_id --format unix)"
echo "Submitted Globus task $TASK_ID"
echo "Progress: https://app.globus.org/activity/$TASK_ID"
if [ "${WAIT:-1}" = "1" ]; then
  globus task wait "$TASK_ID" --polling-interval 15 --heartbeat
  echo
  echo "Done. Next: python scripts/local_inventory.py"
fi
