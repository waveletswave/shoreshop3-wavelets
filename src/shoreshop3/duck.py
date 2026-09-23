"""Duck (FRF) single-profile hindcasts: read ShoreShop3 submissions and FRF observations.

The ShoreShop3 Duck task asks for daily shoreline positions at two FRF profile
lines, yFRF = 1 m (about 500 m south of the pier) and yFRF = 1006 m (about
500 m north), in files named ``mipDuck_<period>_<name>.csv``. Submissions differ
in small ways (header row, date format, NaN spelling, reference baseline); the
readers here normalise all of them to a DataFrame with a daily DatetimeIndex
and one float column per profile ("1", "1006").

Observed shoreline positions come from ``FRF_Profiles.zip``
(``FRF_Profiles/<yFRF:05d>/shorelinePosAtyFRF<yFRF>.csv``, column ``xFRF``),
read directly from the zip so ``data/raw`` stays an untouched mirror.

Baselines differ between submissions (e.g. CoSMoS-COAST uses a different
origin), so compare anomalies: the wavelet tools remove each series' mean.

Surveys come every 2-4 weeks, so the observed series on a regular grid is an
interpolation. By default (``sampling="surveys"``) each model is read on the
survey days and interpolated the same way, so observations and models carry
the same sampling filter and only their differences remain.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .timeseries import regularize

__all__ = [
    "PROFILES",
    "FRF_ZIP",
    "read_submission",
    "read_frf_shoreline",
    "find_submissions",
    "team_folders",
    "load_model_table",
    "DuckCase",
    "build_case",
]

PROFILES = ("1", "1006")
FRF_ZIP = Path("InputData/hindcast_1980_2023/shorelines_and_profiles/FRF_Profiles.zip")
_TEAM_TOPS = ("UserSubmissions", "PublicSubmissions")


def _profile_name(col) -> str | None:
    """'1', '1.0', 1006.0 -> '1' / '1006'; anything non-integer -> None."""
    try:
        v = float(str(col).strip())
    except ValueError:
        return None
    if not np.isfinite(v) or v != int(v):
        return None
    return str(int(v))


def read_submission(path: str | Path) -> pd.DataFrame:
    """Daily shoreline positions per profile from one ``mipDuck_*.csv`` submission.

    First column = dates (ISO or M/D/Y); other columns named by profile
    ('1', '1.0', '1006', ...). Rows without a valid date are dropped, blank or
    'nan' values become NaN, duplicate dates keep the first value.
    """
    df = pd.read_csv(path)
    t = pd.to_datetime(df[df.columns[0]], errors="coerce", format="mixed")
    keep = t.notna().to_numpy()
    cols = {c: _profile_name(c) for c in df.columns[1:]}
    cols = {c: p for c, p in cols.items() if p is not None}
    if not cols:
        raise ValueError(f"{path}: no profile columns found in header {list(df.columns)}")
    out = df.loc[keep, list(cols)].rename(columns=cols).apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(t[keep]).normalize()
    out.index.name = "time"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.astype(float)


def read_frf_shoreline(source: str | Path, profile: str | int) -> pd.Series:
    """Observed shoreline cross-shore position (xFRF, m) at an FRF profile line.

    ``source`` is ``FRF_Profiles.zip``, the folder it was extracted to, or the
    ShoreShop3 mirror root (``data/raw``).
    """
    source = Path(source)
    p = int(float(profile))
    member = f"FRF_Profiles/{p:05d}/shorelinePosAtyFRF{p}.csv"
    if source.is_dir() and (source / FRF_ZIP).exists():
        source = source / FRF_ZIP
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as zf, zf.open(member) as fh:
            df = pd.read_csv(fh)
    else:
        folder = source if (source / member).exists() else source.parent
        df = pd.read_csv(folder / member)
    t = pd.to_datetime(df["time"], errors="coerce", format="mixed")
    s = pd.Series(pd.to_numeric(df["xFRF"], errors="coerce").to_numpy(), index=pd.DatetimeIndex(t),
                  name=f"obs_{p}")
    s = s[s.index.notna() & s.notna()].sort_index()
    return s.groupby(level=0).mean()


def load_model_table(path: str | Path | None = None) -> pd.DataFrame:
    """Model labels and (provisional) families, keyed by submission file name."""
    if path is None:
        from .paths import ROOT

        path = ROOT / "config" / "duck_models.csv"
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["file", "team", "model", "family", "notes"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def find_submissions(raw_root: str | Path, period: str = "1980-2023",
                     model_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """All ``mipDuck_<period>*.csv`` submissions under the mirror, with labels.

    Returns columns: path, file, top, folder, team, model, family. Files
    missing from the model table get the team folder name and the file stem as
    label. A file name found in both UserSubmissions/ and PublicSubmissions/ is
    kept once (the UserSubmissions copy); labels that would clash get the
    team folder appended.
    """
    raw_root = Path(raw_root)
    table = load_model_table() if model_table is None else model_table
    known = table.set_index("file") if len(table) else None
    rows = []
    for top in _TEAM_TOPS:
        for path in sorted((raw_root / top).rglob(f"mipDuck_{period}*.csv")):
            rel = path.relative_to(raw_root)
            folder = rel.parts[1] if len(rel.parts) > 2 else ""
            row = dict(path=str(path), file=path.name, top=top, folder=folder, team=folder,
                       model=path.stem.replace(f"mipDuck_{period}", "").strip("_-"), family="Unclassified")
            if known is not None and path.name in known.index:
                info = known.loc[path.name]
                row.update(team=info["team"] or folder, model=info["model"] or row["model"],
                           family=info["family"] or "Unclassified")
            rows.append(row)
    df = pd.DataFrame(rows, columns=["path", "file", "top", "folder", "team", "model", "family"])
    df = df.drop_duplicates(subset="file", keep="first").reset_index(drop=True)
    clash = df["model"].duplicated(keep=False)
    df.loc[clash, "model"] = df.loc[clash, "model"] + " (" + df.loc[clash, "folder"] + ")"
    return df


def team_folders(raw_root: str | Path) -> list[str]:
    """Team folder names under UserSubmissions/ and PublicSubmissions/ of the mirror."""
    raw_root = Path(raw_root)
    names = {p.name for top in _TEAM_TOPS if (raw_root / top).is_dir()
             for p in (raw_root / top).iterdir() if p.is_dir() and not p.name.startswith(".")}
    return sorted(names, key=str.lower)


@dataclass
class DuckCase:
    """Observations and models for one profile on a common regular grid."""

    profile: str
    time: np.ndarray  #: grid (datetime64)
    dt_days: float
    obs: np.ndarray  #: observations interpolated onto the grid
    gap: np.ndarray  #: grid points inside long observation gaps
    obs_raw: pd.Series  #: the surveys themselves
    models: dict[str, np.ndarray] = field(default_factory=dict)
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)  #: one row per model kept
    skipped: dict[str, str] = field(default_factory=dict)  #: model -> reason
    sampling: str = "surveys"  #: how models were put on the grid (see build_case)


SAMPLING = ("surveys", "mean")


def build_case(raw_root: str | Path, profile: str, start: str, end: str, *, step: str = "7D",
               max_gap: str = "60D", sampling: str = "surveys",
               submissions: pd.DataFrame | None = None) -> DuckCase:
    """Put the FRF observations and every submission for one profile on one grid.

    Observations are linearly interpolated onto a regular grid (``step``) and
    stretches between surveys longer than ``max_gap`` are flagged (``gap``).

    ``sampling`` sets how the daily model output reaches the same grid:

    * ``"surveys"`` (default): read each model on the survey days and
      interpolate exactly like the surveys, so both series carry the same
      sampling filter (an observation operator). Differences then come from
      the model, not from the survey schedule.
    * ``"mean"``: average the daily output over ``step`` (centred window).

    Models without a value on every day needed are skipped (see ``skipped``).
    """
    if sampling not in SAMPLING:
        raise ValueError(f"sampling must be one of {SAMPLING}")
    obs_raw = read_frf_shoreline(raw_root, profile)
    reg = regularize(obs_raw.index, obs_raw.to_numpy(), step, max_gap=max_gap, start=start, end=end)
    grid = pd.DatetimeIndex(reg.time)
    # surveys that shape the grid values: the window plus the enclosing survey on each side
    t = obs_raw.index
    i0 = max(int(np.searchsorted(t, grid[0], side="right")) - 1, 0)
    i1 = min(int(np.searchsorted(t, grid[-1], side="left")), t.size - 1)
    t_used = t[i0:i1 + 1]
    subs = find_submissions(raw_root) if submissions is None else submissions
    win = int(round(pd.Timedelta(step) / pd.Timedelta("1D")))
    models, kept, skipped = {}, [], {}
    for _, row in subs.iterrows():
        try:
            df = read_submission(row["path"])
        except (ValueError, KeyError, UnicodeDecodeError, pd.errors.ParserError) as err:
            skipped[row["model"]] = f"could not read {row['file']}: {err}"
            continue
        if profile not in df.columns:
            skipped[row["model"]] = f"no column for profile {profile}"
            continue
        daily = df[profile].asfreq("1D")
        if sampling == "surveys":
            at_surveys = daily.reindex(t_used.normalize()).to_numpy(float)
            if np.isnan(at_surveys).any():
                skipped[row["model"]] = (f"no value on {int(np.isnan(at_surveys).sum())} of {t_used.size} "
                                         f"survey days in {start}..{end}")
                continue
            vals = regularize(t_used, at_surveys, step, max_gap=max_gap, start=start, end=end).values
        else:
            smooth = daily.rolling(win, center=True, min_periods=max(1, win // 2 + 1)).mean() if win > 1 else daily
            vals = smooth.reindex(grid).to_numpy(float)
            if np.isnan(vals).any():
                skipped[row["model"]] = f"{int(np.isnan(vals).sum())} of {vals.size} grid points missing in {start}..{end}"
                continue
        if row["model"] in models:
            raise ValueError(f"duplicate model label {row['model']!r}; make labels unique in config/duck_models.csv")
        models[row["model"]] = vals
        kept.append(row)
    meta = pd.DataFrame(kept).reset_index(drop=True)
    return DuckCase(profile=str(profile), time=reg.time, dt_days=reg.dt, obs=reg.values, gap=reg.gap,
                    obs_raw=obs_raw, models=models, meta=meta, skipped=skipped, sampling=sampling)
