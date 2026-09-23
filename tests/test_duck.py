import zipfile

import numpy as np
import pandas as pd
import pytest

from shoreshop3 import duck


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_read_submission_header_date_and_nan_variants(tmp_path):
    f = _write(tmp_path / "a.csv",
               "Date,1.0,1006.0,notes\n"
               "1/3/1980,10,20,x\n"
               "1/1/1980,nan,21,x\n"
               "1/2/1980,12,,x\n"
               "1/2/1980,99,99,x\n"   # duplicate day: first value kept
               "not a date,1,1,x\n")
    df = duck.read_submission(f)
    assert list(df.columns) == ["1", "1006"]
    assert list(df.index) == list(pd.to_datetime(["1980-01-01", "1980-01-02", "1980-01-03"]))
    assert np.isnan(df.loc["1980-01-01", "1"]) and df.loc["1980-01-01", "1006"] == 21
    assert df.loc["1980-01-02", "1"] == 12 and np.isnan(df.loc["1980-01-02", "1006"])
    assert df.dtypes.eq(float).all()


def test_read_submission_iso_times_are_normalised(tmp_path):
    f = _write(tmp_path / "b.csv", "time,1,1006\n1980-01-01 12:00:00,1,2\n1980-01-02T00:00:00,3,4\n")
    df = duck.read_submission(f)
    assert df.index[0] == pd.Timestamp("1980-01-01")
    assert df.loc["1980-01-02", "1006"] == 4


def test_read_submission_without_profiles_raises(tmp_path):
    f = _write(tmp_path / "c.csv", "time,north,south\n1980-01-01,1,2\n")
    with pytest.raises(ValueError, match="no profile columns"):
        duck.read_submission(f)


def _frf_csv(times, values):
    rows = "\n".join(f"{t},{v},1.0" for t, v in zip(times, values))
    return "time,xFRF,yFRF\n" + rows + "\n"


def _write_frf_zip(raw_root, series: dict[str, tuple]):
    path = raw_root / duck.FRF_ZIP
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for p, (times, values) in series.items():
            zf.writestr(f"FRF_Profiles/{int(p):05d}/shorelinePosAtyFRF{p}.csv", _frf_csv(times, values))
    return path


def test_read_frf_shoreline_from_zip_folder_and_root(tmp_path):
    times = ["2000-01-02 10:00:00", "2000-01-01 09:00:00", "2000-01-02 10:00:00", "2000-01-05 08:00:00"]
    zpath = _write_frf_zip(tmp_path / "raw", {"1": (times, [3.0, 1.0, 5.0, 7.0])})
    s_root = duck.read_frf_shoreline(tmp_path / "raw", "1")
    s_zip = duck.read_frf_shoreline(zpath, 1)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(tmp_path / "extracted")
    s_dir = duck.read_frf_shoreline(tmp_path / "extracted", "1.0")
    for s in (s_root, s_zip, s_dir):
        assert s.index.is_monotonic_increasing
        assert s.tolist() == [1.0, 4.0, 7.0]  # duplicate time stamps averaged


def _duck_mirror(tmp_path, *, gap=True):
    """Mirror with 10-daily surveys at profile 1 and three kinds of submissions."""
    raw = tmp_path / "raw"
    days = pd.date_range("2000-01-01", "2003-12-31", freq="10D")
    if gap:
        days = days[(days < "2001-06-01") | (days > "2001-09-15")]  # ~100-day survey gap
    t = days + pd.Timedelta("13h")
    truth = lambda d: 50 + 10 * np.sin(2 * np.pi * (d - pd.Timestamp("2000-01-01")).days / 365.25)  # noqa: E731
    _write_frf_zip(raw, {"1": ([str(x) for x in t], truth(days).round(6).tolist()),
                         "1006": ([str(x) for x in t], truth(days).round(6).tolist())})
    daily = pd.date_range("1999-06-01", "2004-06-30", freq="D")
    full = pd.DataFrame({"time": daily.strftime("%Y-%m-%d"), "1": truth(daily).round(6),
                         "1006": truth(daily).round(6) + 3})
    team = raw / "UserSubmissions" / "TeamA" / "Duck"
    team.mkdir(parents=True)
    full.to_csv(team / "mipDuck_1980-2023_perfect.csv", index=False)
    holes = full.copy()
    holes.loc[(daily > "2002-01-01") & (daily < "2002-03-01"), "1"] = np.nan
    holes.to_csv(team / "mipDuck_1980-2023_holes.csv", index=False)
    pub = raw / "PublicSubmissions" / "TeamB"
    pub.mkdir(parents=True)
    full[["time", "1006"]].to_csv(pub / "mipDuck_1980-2023-northonly.csv", index=False)
    full.to_csv(pub / "mipDuck_calibration_other.csv", index=False)  # other period: ignored
    (raw / "UserSubmissions" / "TeamC").mkdir()  # a team without Duck files
    return raw


def test_find_submissions_labels_and_team_folders(tmp_path):
    raw = _duck_mirror(tmp_path)
    table = pd.DataFrame([dict(file="mipDuck_1980-2023_perfect.csv", team="Team A", model="Perfect",
                               family="Equilibrium", notes="")])
    subs = duck.find_submissions(raw, model_table=table)
    assert len(subs) == 3
    row = subs.set_index("file").loc["mipDuck_1980-2023_perfect.csv"]
    assert (row["team"], row["model"], row["family"], row["folder"]) == ("Team A", "Perfect", "Equilibrium", "TeamA")
    other = subs.set_index("file").loc["mipDuck_1980-2023-northonly.csv"]
    assert (other["team"], other["model"], other["family"]) == ("TeamB", "northonly", "Unclassified")
    assert duck.team_folders(raw) == ["TeamA", "TeamB", "TeamC"]


def test_find_submissions_duplicates_clashes_and_bad_files(tmp_path):
    raw = _duck_mirror(tmp_path, gap=False)
    user_copy = raw / "UserSubmissions" / "TeamB" / "mipDuck_1980-2023-northonly.csv"
    user_copy.parent.mkdir(parents=True)
    user_copy.write_text((raw / "PublicSubmissions" / "TeamB" / "mipDuck_1980-2023-northonly.csv").read_text())
    _write(raw / "UserSubmissions" / "TeamC" / "mipDuck_1980-2023_perfect.csv.csv", "x")  # other file name
    _write(raw / "UserSubmissions" / "TeamD" / "mipDuck_1980-2023-perfect.csv", "time,north\n2000-01-01,1\n")
    subs = duck.find_submissions(raw, model_table=duck.load_model_table(tmp_path / "missing.csv"))
    assert subs["file"].is_unique
    assert subs.set_index("file").loc["mipDuck_1980-2023-northonly.csv", "top"] == "UserSubmissions"
    assert {"perfect (TeamA)", "perfect (TeamD)"} <= set(subs["model"])
    case = duck.build_case(raw, "1006", "2000-02-01", "2003-11-30", submissions=subs)
    assert "could not read" in case.skipped["perfect (TeamD)"]
    assert "could not read" in case.skipped["perfect.csv"]
    assert "perfect (TeamA)" in case.models


@pytest.mark.parametrize("sampling", duck.SAMPLING)
def test_build_case_grid_gaps_and_skips(tmp_path, sampling):
    raw = _duck_mirror(tmp_path)
    subs = duck.find_submissions(raw, model_table=duck.load_model_table(tmp_path / "missing.csv"))
    case = duck.build_case(raw, "1", "2000-02-01", "2003-11-30", step="7D", max_gap="60D",
                           sampling=sampling, submissions=subs)
    assert case.dt_days == 7.0 and case.sampling == sampling
    assert np.issubdtype(case.time.dtype, np.datetime64)
    t = pd.DatetimeIndex(case.time)
    assert case.gap[(t > "2001-06-15") & (t < "2001-09-01")].all()
    assert not case.gap[t < "2001-05-01"].any()
    assert set(case.models) == {"perfect"}
    assert "holes" in case.skipped and "northonly" in case.skipped
    assert list(case.meta["model"]) == ["perfect"]
    err = np.abs(case.models["perfect"] - case.obs)
    if sampling == "surveys":
        assert err.max() < 1e-6  # same days, same interpolation: identical series
    else:
        assert 0 < err[~case.gap].max() < 0.5  # weekly mean vs interpolated surveys


def test_build_case_north_profile_and_duplicate_labels(tmp_path):
    raw = _duck_mirror(tmp_path, gap=False)
    subs = duck.find_submissions(raw, model_table=duck.load_model_table(tmp_path / "missing.csv"))
    case = duck.build_case(raw, "1006", "2000-02-01", "2003-11-30", submissions=subs)
    assert set(case.models) == {"perfect", "holes", "northonly"}
    assert not case.gap.any()
    assert np.allclose(case.models["perfect"] - case.obs, 3.0)  # offsets survive; anomalies are compared
    dup = subs.assign(model="same")
    with pytest.raises(ValueError, match="duplicate model label"):
        duck.build_case(raw, "1006", "2000-02-01", "2003-11-30", submissions=dup)
    with pytest.raises(ValueError, match="sampling"):
        duck.build_case(raw, "1006", "2000-02-01", "2003-11-30", sampling="nearest", submissions=subs)
