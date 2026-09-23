#!/usr/bin/env python3
"""Inventory of the local ShoreShop3 mirror (data/raw).

Lists every file per team, peeks inside CSV / NetCDF / MAT / ZIP / JSON files
(columns, dimensions, variables, time range) and checks which template file
names from SubmissionTemplates each team used. Writes to outputs/inventory/:

  INVENTORY.md   readable overview (share this with Brad / collaborators)
  teams.csv      one row per team: file count, size, file types, templates found
  files.csv      one row per file, with a one-line description of its contents
  peek.jsonl     full details per file (columns, variables, dims, ...)

Usage: python scripts/local_inventory.py [--raw data/raw] [--out outputs/inventory]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # usable before `pip install -e .`

import pandas as pd  # noqa: E402

from shoreshop3.inventory import build_inventory  # noqa: E402
from shoreshop3.paths import OUTPUTS, RAW  # noqa: E402  (honours SHORESHOP3_DATA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "inventory")
    ap.add_argument("--no-peek", action="store_true", help="only list files, do not open them")
    ap.add_argument("--max-peek-mb", type=float, default=500.0,
                    help="skip peeking into files larger than this")
    args = ap.parse_args()

    if not args.raw.exists() or not any(p for p in args.raw.iterdir() if not p.name.startswith(".")):
        sys.exit(f"{args.raw} is empty: run scripts/globus_sync.sh first")
    files, teams = build_inventory(args.raw, args.out, do_peek=not args.no_peek,
                                   max_peek_mb=args.max_peek_mb)
    with pd.option_context("display.max_rows", 500, "display.width", 160,
                           "display.max_colwidth", 60):
        print(teams.to_string(index=False))
    print(f"\n{len(files)} files, {files['size_bytes'].sum() / 1e9:.2f} GB")
    print(f"Wrote {args.out / 'INVENTORY.md'} (+ teams.csv, files.csv, peek.jsonl)")


if __name__ == "__main__":
    main()
