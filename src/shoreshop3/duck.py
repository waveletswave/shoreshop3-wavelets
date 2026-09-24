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
    "coverage",
    "coverage_problem",
    "needed_days",
    "WINDOW_MODES",
    "Window",
    "choose_window",
    "common_window",
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


def coverage(submissions: pd.DataFrame, profiles=PROFILES) -> pd.DataFrame:
    """First and last day with a value, per submission and profile (unreadable files left out)."""
    rows = []
    for r in submissions.itertuples():
        try:
            df = read_submission(r.path)
        except (ValueError, KeyError, UnicodeDecodeError, pd.errors.ParserError):
            continue
        for p in profiles:
            if str(p) in df.columns:
                s = df[str(p)].dropna()
                if len(s):
                    rows.append(dict(model=r.model, profile=str(p), first=s.index.min(), last=s.index.max()))
    return pd.DataFrame(rows, columns=["model", "profile", "first", "last"])


WINDOW_MODES = ("surveys", "common")
_DAY = pd.Timedelta("1D")


def needed_days(t: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    """Survey times whose values enter a grid over [start, end].

    Those inside the window plus the enclosing survey on each side (linear
    interpolation of the first and last grid points uses them).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    i0 = max(int(np.searchsorted(t, start, side="right")) - 1, 0)
    i1 = min(int(np.searchsorted(t, end, side="left")), t.size - 1)
    return t[i0:i1 + 1]


def coverage_problem(daily: pd.Series, days) -> str | None:
    """Why a daily model series cannot be read on ``days``; None if it can.

    ``days`` are the survey days that enter the grid (see :func:`needed_days`),
    so the first one can lie a little before the window start.
    """
    days = pd.DatetimeIndex(days).normalize()
    vals = daily.reindex(days)
    if not vals.isna().any():
        return None
    s = daily.dropna()
    if s.empty:
        return "no values"
    if s.index.min() > days.min():
        return f"starts {s.index.min().date()}, after the first day needed ({days.min().date()})"
    if s.index.max() < days.max():
        return f"ends {s.index.max().date()}, before the last day needed ({days.max().date()})"
    return f"no value on {int(vals.isna().sum())} of {len(days)} days needed inside the window"


@dataclass
class Window:
    """Analysis window and what set its limits."""

    start: pd.Timestamp
    end: pd.Timestamp
    mode: str
    start_set_by: str
    end_set_by: str
    #: "model at profile p" -> reason, for models set aside while fixing a common window
    set_aside: dict[str, str] = field(default_factory=dict)


def choose_window(raw_root: str | Path, profiles=PROFILES, submissions: pd.DataFrame | None = None,
                  mode: str = "surveys", start: str | None = None, end: str | None = None) -> Window:
    """Analysis window for the given profiles.

    ``mode="surveys"`` (default): the whole survey record shared by the
    profiles, from the day after the first survey to the day of the last one.
    Models that do not cover it take no part; :func:`build_case` says why.

    ``mode="common"``: the longest window covered by the surveys and by every
    model that can take part. The models are checked first: one that lacks a
    value on a day the grid needs is set aside and the window is recomputed
    without it, until nobody is set aside. A model that cannot take part
    therefore never shortens the window for the others.

    ``start`` / ``end`` fix either limit by hand.
    """
    if mode not in WINDOW_MODES:
        raise ValueError(f"mode must be one of {WINDOW_MODES}")
    fixed_s = None if start is None else pd.Timestamp(start)
    fixed_e = None if end is None else pd.Timestamp(end)
    surveys = {str(p): read_frf_shoreline(raw_root, p).index for p in profiles}
    s_surv = max(t[0].normalize() + _DAY for t in surveys.values())
    e_surv = min(t[-1].normalize() for t in surveys.values())
    if mode == "surveys":
        s = s_surv if fixed_s is None else fixed_s
        e = e_surv if fixed_e is None else fixed_e
        if e <= s:
            raise ValueError(f"empty window: {s.date()} to {e.date()}")
        return Window(s, e, mode, "set by hand" if fixed_s is not None else "the first surveys",
                      "set by hand" if fixed_e is not None else "the last surveys")

    subs = find_submissions(raw_root) if submissions is None else submissions
    active = {}
    for r in subs.itertuples():
        try:
            df = read_submission(r.path)
        except (ValueError, KeyError, UnicodeDecodeError, pd.errors.ParserError):
            continue
        for p in surveys:
            if p in df.columns and df[p].notna().any():
                active[(r.model, p)] = df[p].asfreq("1D")
    set_aside = {}
    while True:
        if not active:
            raise ValueError("no submission can take part in any window")
        m_start = max(v.first_valid_index() for v in active.values())
        m_end = min(v.last_valid_index() for v in active.values())
        s_by_p, e_by_p = {}, {}
        for p, t in surveys.items():
            after, before = t[t >= m_start], t[t < m_end.normalize() + _DAY]
            if not len(after) or not len(before):
                raise ValueError(f"the surveys at profile {p} do not overlap the submissions")
            s_by_p[p] = after[0].normalize() + _DAY
            e_by_p[p] = before[-1].normalize()
        s = max(s_by_p.values()) if fixed_s is None else fixed_s
        e = min(e_by_p.values()) if fixed_e is None else fixed_e
        if e <= s:
            raise ValueError(f"empty window: {s.date()} to {e.date()}")
        failed = {k: coverage_problem(v, needed_days(surveys[k[1]], s, e)) for k, v in active.items()}
        failed = {k: why for k, why in failed.items() if why}
        if not failed:
            break
        for (m, p), why in failed.items():
            set_aside[f"{m} at profile {p}"] = why
            del active[(m, p)]

    def _names(which: str, value) -> str:
        keys = [k for k, v in active.items()
                if (v.first_valid_index() if which == "first" else v.last_valid_index()) == value]
        return ", ".join(f"{m} at profile {p}" for m, p in keys)

    if fixed_s is not None:
        start_by = "set by hand"
    else:
        p_bind = max(s_by_p, key=s_by_p.get)
        start_by = "the first surveys" if surveys[p_bind][0] >= m_start else _names("first", m_start)
    if fixed_e is not None:
        end_by = "set by hand"
    else:
        p_bind = min(e_by_p, key=e_by_p.get)
        end_by = "the last surveys" if surveys[p_bind][-1] < m_end.normalize() + _DAY else _names("last", m_end)
    return Window(s, e, mode, start_by, end_by, set_aside)


def common_window(raw_root: str | Path, profiles=PROFILES, submissions: pd.DataFrame | None = None,
                  start: str | None = None, end: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp, dict]:
    """:func:`choose_window` with ``mode="common"``, returned as (start, end, info)."""
    w = choose_window(raw_root, profiles, submissions, mode="common", start=start, end=end)
    info = {"start_set_by": "--start" if start is not None else w.start_set_by,
            "end_set_by": "--end" if end is not None else w.end_set_by, "set_aside": w.set_aside}
    return w.start, w.end, info


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
    t_used = needed_days(obs_raw.index, grid[0], grid[-1])
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
            why = coverage_problem(daily, t_used)
            if why:
                skipped[row["model"]] = why
                continue
            at_surveys = daily.reindex(t_used.normalize()).to_numpy(float)
            vals = regularize(t_used, at_surveys, step, max_gap=max_gap, start=start, end=end).values
        else:
            smooth = daily.rolling(win, center=True, min_periods=max(1, win // 2 + 1)).mean() if win > 1 else daily
            why = coverage_problem(smooth, grid)
            if why:
                skipped[row["model"]] = why
                continue
            vals = smooth.reindex(grid).to_numpy(float)
        if row["model"] in models:
            raise ValueError(f"duplicate model label {row['model']!r}; make labels unique in config/duck_models.csv")
        models[row["model"]] = vals
        kept.append(row)
    meta = pd.DataFrame(kept).reset_index(drop=True)
    return DuckCase(profile=str(profile), time=reg.time, dt_days=reg.dt, obs=reg.values, gap=reg.gap,
                    obs_raw=obs_raw, models=models, meta=meta, skipped=skipped, sampling=sampling)
