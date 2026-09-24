"""Inventory of the ShoreShop3 data: what each team submitted, in which format.

Two sources are supported:

* the local mirror in ``data/raw`` (after ``scripts/globus_sync.sh``):
  :func:`build_inventory` lists every file, peeks inside CSV / NetCDF / MAT /
  ZIP files, and writes ``files.csv``, ``teams.csv``, ``peek.jsonl`` and a
  readable ``INVENTORY.md``;
* the remote listing written by ``scripts/globus_manifest.sh``
  (``globus ls -r -F json``): :func:`load_manifest` and
  :func:`summarize_files` give sizes per team before anything is downloaded.
"""

from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

__all__ = [
    "TEAM_FOLDERS",
    "scan",
    "peek",
    "summarize_files",
    "load_manifest",
    "build_inventory",
]

#: top-level folders whose sub-folders are model teams
TEAM_FOLDERS = ("UserSubmissions", "PublicSubmissions")
TEMPLATE_FOLDER = "SubmissionTemplates"

_TEXT_EXT = {".csv", ".tsv", ".txt", ".dat", ".asc"}
_NC_EXT = {".nc", ".nc4", ".cdf", ".netcdf"}
_COLUMNS = ["top", "team", "relpath", "name", "ext", "size_bytes", "modified"]


def _ext(name: str) -> str:
    low = name.lower()
    for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if low.endswith(compound):
            return compound
    return os.path.splitext(low)[1]


def _top_team(parts: tuple[str, ...]) -> tuple[str, str]:
    top = parts[0] if len(parts) > 1 else ""
    team = parts[1] if top in TEAM_FOLDERS and len(parts) > 2 else ""
    return top, team


def scan(raw_root: str | Path) -> pd.DataFrame:
    """One row per file under ``raw_root`` (hidden files are skipped)."""
    root = Path(raw_root)
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            path = Path(dirpath) / fname
            rel = path.relative_to(root)
            top, team = _top_team(rel.parts)
            st = path.stat()
            rows.append(dict(top=top, team=team, relpath=rel.as_posix(), name=fname,
                             ext=_ext(fname), size_bytes=st.st_size,
                             modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")))
    return pd.DataFrame(rows, columns=_COLUMNS)


# --- peeking inside files ------------------------------------------------------------

def _count_lines(path: Path) -> int:
    n = 0
    last = b"\n"
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n + (0 if last == b"\n" else 1)


def _last_line(path: Path) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 65536))
        tail = fh.read().decode("utf-8", "replace").rstrip("\r\n")
    return tail.splitlines()[-1] if tail else ""


def _peek_text(path: Path) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        head = [fh.readline() for _ in range(6)]
    head = [h for h in head if h]
    sample = "".join(head)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    rows = list(csv.reader(io.StringIO(sample), delimiter=delim, skipinitialspace=True))
    header = rows[0] if rows else []
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = bool(header) and not _looks_numeric(header[0])
    n_lines = _count_lines(path)
    first = rows[1] if has_header and len(rows) > 1 else (rows[0] if rows else [])
    last = next(csv.reader([_last_line(path)], delimiter=delim, skipinitialspace=True), [])
    info = dict(kind="table", delimiter=delim, has_header=has_header,
                n_rows=n_lines - (1 if has_header else 0), n_cols=len(header),
                columns=header[:40] if has_header else [],
                first_row=first[:6], last_row=last[:6])
    t0, t1 = _as_date(first[0] if first else None), _as_date(last[0] if last else None)
    if t0 and t1:
        info["time_range"] = [t0, t1]
    cols = ", ".join(info["columns"][:8]) + (" ..." if info["n_cols"] > 8 else "")
    rng = f"; {t0} -> {t1}" if t0 and t1 else ""
    info["summary"] = f"{info['n_rows']} rows x {info['n_cols']} cols [{cols}]{rng}"
    return info


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _as_date(s: str | None) -> str | None:
    if not s or _looks_numeric(s):
        return None
    try:
        ts = pd.Timestamp(s)
    except (ValueError, TypeError, OverflowError):
        return None
    return None if pd.isna(ts) else ts.isoformat()


def _peek_netcdf(path: Path) -> dict:
    import netCDF4  # imported lazily: optional for the rest of the module

    with netCDF4.Dataset(path) as ds:
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        variables = {k: dict(dims=list(v.dimensions), units=getattr(v, "units", ""),
                             long_name=getattr(v, "long_name", "")) for k, v in ds.variables.items()}
        info = dict(kind="netcdf", dims=dims, variables=variables, attrs=list(ds.ncattrs())[:30])
        tname = next((k for k in ("time", "Time", "t", "date") if k in ds.variables), None)
        if tname is not None and ds.variables[tname].size:
            tv = ds.variables[tname]
            try:
                vals = netCDF4.num2date(tv[[0, -1]], tv.units, getattr(tv, "calendar", "standard"))
                info["time_range"] = [str(vals[0]), str(vals[1])]
            except Exception:  # noqa: BLE001 - units missing or unusual calendars
                info["time_range"] = [str(tv[0]), str(tv[-1])]
    data_vars = [k for k in variables if k not in dims]
    rng = f"; {info['time_range'][0]} -> {info['time_range'][1]}" if "time_range" in info else ""
    info["summary"] = f"dims {dims}; vars {data_vars[:10]}{rng}"
    return info


def _peek_mat(path: Path) -> dict:
    from scipy.io import whosmat

    try:
        items = [(n, list(s), c) for n, s, c in whosmat(path)]
        info = dict(kind="mat", variables=items)
    except (NotImplementedError, ValueError):  # MATLAB v7.3 files are HDF5
        import h5py

        items = []
        with h5py.File(path, "r") as f:
            f.visititems(lambda n, o: items.append((n, list(o.shape), str(o.dtype)))
                         if isinstance(o, h5py.Dataset) else None)
        info = dict(kind="mat-v7.3", variables=items[:200])
    info["summary"] = "; ".join(f"{n} {tuple(s)}" for n, s, _ in items[:10])
    return info


def _peek_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        total = sum(m.file_size for m in members)
    info = dict(kind="zip", n_members=len(members), uncompressed_bytes=total,
                members=[m.filename for m in members[:50]])
    info["summary"] = f"{len(members)} files, {total / 1e6:.1f} MB uncompressed"
    return info


def _peek_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    keys = list(obj)[:30] if isinstance(obj, dict) else []
    info = dict(kind="json", top_type=type(obj).__name__, keys=keys,
                length=len(obj) if hasattr(obj, "__len__") else None)
    info["summary"] = f"{info['top_type']} keys={keys[:10]}" if keys else f"{info['top_type']}"
    return info


def peek(path: str | Path) -> dict:
    """Describe a file's contents without loading it fully. Never raises."""
    path = Path(path)
    ext = _ext(path.name)
    try:
        if ext in _TEXT_EXT:
            return _peek_text(path)
        if ext in _NC_EXT:
            return _peek_netcdf(path)
        if ext == ".mat":
            return _peek_mat(path)
        if ext == ".zip":
            return _peek_zip(path)
        if ext == ".json":
            return _peek_json(path)
        if ext in {".md", ".rst"}:
            with open(path, encoding="utf-8", errors="replace") as fh:
                first = next((ln.strip() for ln in fh if ln.strip()), "")
            return dict(kind="text", summary=first[:120])
    except Exception as exc:  # noqa: BLE001 - report instead of crashing the inventory
        return dict(kind="error", summary=f"could not read: {type(exc).__name__}: {exc}"[:200])
    return dict(kind="other", summary="")


# --- summaries ---------------------------------------------------------------------------

def _ext_counts(exts: Iterable[str]) -> str:
    return ", ".join(f"{e or '(none)'}:{c}" for e, c in Counter(exts).most_common())


def summarize_files(files: pd.DataFrame) -> pd.DataFrame:
    """Per (top, team): file count, total size (MB) and extension counts."""
    if files.empty:
        return pd.DataFrame(columns=["top", "team", "n_files", "total_mb", "extensions"])
    g = files.groupby(["top", "team"], dropna=False, sort=True)
    out = g.agg(n_files=("relpath", "size"), total_mb=("size_bytes", "sum"),
                extensions=("ext", _ext_counts)).reset_index()
    out["total_mb"] = (out["total_mb"] / 1e6).round(3)
    return out


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read one ``globus ls -r -F json`` listing (file name = top-level folder).

    Returns the same columns as :func:`scan` plus ``type``.
    """
    path = Path(path)
    top = path.stem
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh).get("DATA", [])
    rows = []
    for it in items:
        rel = str(it.get("name", "")).strip("/")
        typ = it.get("type", "file")
        parts = rel.split("/")
        if top in TEAM_FOLDERS:
            team = parts[0] if (len(parts) > 1 or typ == "dir") else ""
        else:
            team = ""
        rows.append(dict(top=top, team=team, relpath=f"{top}/{rel}", name=parts[-1],
                         ext=_ext(parts[-1]) if typ == "file" else "",
                         size_bytes=int(it.get("size") or 0),
                         modified=it.get("last_modified"), type=typ))
    return pd.DataFrame(rows, columns=_COLUMNS + ["type"])


def _template_names(files: pd.DataFrame) -> set[str]:
    return set(files.loc[files["top"] == TEMPLATE_FOLDER, "name"])


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def build_inventory(raw_root: str | Path, out_dir: str | Path, *, do_peek: bool = True,
                    max_peek_mb: float = 500.0, max_files_listed: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan ``raw_root`` and write files.csv, teams.csv, peek.jsonl, INVENTORY.md."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = scan(raw_root)
    root = Path(raw_root)

    summaries = []
    with open(out / "peek.jsonl", "w", encoding="utf-8") as fh:
        for _, row in files.iterrows():
            if not do_peek:
                summaries.append("")
                continue
            if row["size_bytes"] > max_peek_mb * 1e6:
                info = dict(kind="skipped", summary=f"larger than {max_peek_mb:g} MB")
            else:
                info = peek(root / row["relpath"])
            summaries.append(info.get("summary", ""))
            fh.write(json.dumps(dict(relpath=row["relpath"], **info), default=str) + "\n")
    files = files.assign(peek=summaries)
    files.to_csv(out / "files.csv", index=False)

    teams = summarize_files(files)
    templates = _template_names(files)
    if templates:
        found = files[files["team"] != ""].groupby(["top", "team"])["name"].agg(
            lambda s: len(templates & set(s)))
        teams = teams.merge(found.rename("template_files_found").reset_index(),
                            on=["top", "team"], how="left")
        teams["template_files_found"] = [
            int(v) if team else "" for team, v in
            zip(teams["team"], teams["template_files_found"].fillna(0))]
    teams.to_csv(out / "teams.csv", index=False)

    md = ["# ShoreShop3 data inventory\n",
          f"Source: `{root}` - scanned {datetime.now().isoformat(timespec='minutes')}\n",
          f"{len(files)} files, {files['size_bytes'].sum() / 1e9:.2f} GB in total.\n"]
    if templates:
        md.append(f"{len(templates)} template file names in `{TEMPLATE_FOLDER}/`; "
                  "`template_files_found` counts how many of them each team has.\n")
    md += ["## Overview\n", _markdown_table(teams), ""]
    for (top, team), grp in files.groupby(["top", "team"], sort=True):
        title = f"{top}/{team}" if team else (top or "(root)")
        md.append(f"## {title}\n")
        md.append(f"{len(grp)} file{'s' if len(grp) != 1 else ''}, "
                  f"{grp['size_bytes'].sum() / 1e6:.1f} MB - {_ext_counts(grp['ext'])}\n")
        listed = pd.DataFrame({
            "file": grp["relpath"].head(max_files_listed).map(lambda p: f"`{p}`"),
            "MB": (grp["size_bytes"].head(max_files_listed) / 1e6).round(3),
            "contents": grp["peek"].head(max_files_listed),
        })
        md.append(_markdown_table(listed))
        if len(grp) > max_files_listed:
            md.append(f"\n... and {len(grp) - max_files_listed} more (see files.csv)")
        md.append("")
    (out / "INVENTORY.md").write_text("\n".join(md), encoding="utf-8")
    return files, teams
