"""Project paths, so notebooks and scripts work from any working directory.

Set the environment variable ``SHORESHOP3_ROOT`` to override the project root,
or ``SHORESHOP3_DATA`` to keep the (large) data directory somewhere else, e.g.
on an external drive or the Duke cluster.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("SHORESHOP3_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for parent in [*here.parents, cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "shoreshop3").is_dir():
            return parent
    return Path.cwd().resolve()


ROOT = _find_root()
DATA = Path(os.environ.get("SHORESHOP3_DATA", ROOT / "data")).expanduser()
RAW = DATA / "raw"  #: read-only mirror of the Globus collection
MANIFESTS = DATA / "manifests"  #: remote listings from globus_manifest.sh
INTERIM = DATA / "interim"  #: caches, intermediate products
PROCESSED = DATA / "processed"  #: harmonised model x transect x time datasets
OUTPUTS = ROOT / "outputs"  #: figures and tables
