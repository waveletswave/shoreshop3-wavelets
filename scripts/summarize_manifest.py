#!/usr/bin/env python3
"""Summarise the remote listings written by globus_manifest.sh.

Prints file counts, sizes and file types per team (nothing is downloaded) and
writes summary.csv next to the JSON listings.

Usage: python scripts/summarize_manifest.py [data/manifests/<timestamp>]
       (default: the most recent listing)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # usable before `pip install -e .`

import pandas as pd  # noqa: E402

from shoreshop3.inventory import load_manifest, summarize_files  # noqa: E402


def latest_manifest(base: Path) -> Path:
    runs = sorted(p for p in base.glob("*") if p.is_dir() and any(p.glob("*.json")))
    if not runs:
        sys.exit(f"No listings in {base}: run scripts/globus_manifest.sh first")
    return runs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest_dir", nargs="?", type=Path)
    args = ap.parse_args()
    folder = args.manifest_dir or latest_manifest(ROOT / "data" / "manifests")

    listing = pd.concat([load_manifest(p) for p in sorted(folder.glob("*.json"))], ignore_index=True)
    files = listing[listing["type"] == "file"]
    summary = summarize_files(files)
    summary.to_csv(folder / "summary.csv", index=False)

    with pd.option_context("display.max_rows", 500, "display.width", 160,
                           "display.max_colwidth", 60):
        print(summary.to_string(index=False))
    teams = summary.loc[summary["team"] != "", "team"].nunique()
    print(f"\n{len(files)} files, {files['size_bytes'].sum() / 1e9:.2f} GB, {teams} team folders")
    print(f"Saved {folder / 'summary.csv'}")


if __name__ == "__main__":
    main()
