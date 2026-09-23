import json

import numpy as np
import pandas as pd
import pytest

from shoreshop3 import inventory as inv


@pytest.fixture
def fake_raw(tmp_path):
    raw = tmp_path / "raw"
    (raw / "SubmissionTemplates").mkdir(parents=True)
    (raw / "SubmissionTemplates" / "shorelines_prediction.csv").write_text("Datetime,T1,T2\n")
    (raw / "InputData").mkdir()
    (raw / "InputData" / "waves.csv").write_text(
        "Datetime,Hs,Tp\n2000-01-01,1.0,8\n2000-01-02,1.2,9\n2000-01-03,0.9,7\n")
    team_a = raw / "UserSubmissions" / "TEAM_A"
    team_a.mkdir(parents=True)
    (team_a / "shorelines_prediction.csv").write_text(
        "Datetime,T1,T2\n2019-01-01,10,11\n2019-01-02,10.5,11.2\n")
    (team_a / "README.md").write_text("# Team A model\nOne-line model.\n")
    team_b = raw / "UserSubmissions" / "TEAM_B" / "runs"
    team_b.mkdir(parents=True)
    (team_b / "output.txt").write_text("1 2 3\n4 5 6\n")
    (team_b / ".DS_Store").write_text("x")
    (raw / "PublicSubmissions" / "TEAM_C").mkdir(parents=True)
    (raw / "PublicSubmissions" / "TEAM_C" / "result.json").write_text(json.dumps({"a": 1, "b": 2}))
    return raw


def test_scan_assigns_top_and_team(fake_raw):
    files = inv.scan(fake_raw)
    assert ".DS_Store" not in set(files["name"])
    row = files.set_index("relpath").loc["UserSubmissions/TEAM_B/runs/output.txt"]
    assert row["top"] == "UserSubmissions" and row["team"] == "TEAM_B" and row["ext"] == ".txt"
    assert set(files.loc[files.top == "InputData", "team"]) == {""}


def test_peek_csv_reads_header_rows_and_dates(fake_raw):
    info = inv.peek(fake_raw / "InputData" / "waves.csv")
    assert info["kind"] == "table"
    assert info["columns"] == ["Datetime", "Hs", "Tp"]
    assert info["n_rows"] == 3
    assert info["time_range"][0].startswith("2000-01-01")
    assert info["time_range"][1].startswith("2000-01-03")


def test_peek_netcdf(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.Dataset({"shoreline": (("time", "transect"), np.zeros((3, 2)))},
                    coords={"time": pd.date_range("2010-01-01", periods=3), "transect": [1, 2]})
    path = tmp_path / "x.nc"
    ds.to_netcdf(path, engine="netcdf4")
    info = inv.peek(path)
    assert info["kind"] == "netcdf"
    assert info["dims"] == {"time": 3, "transect": 2}
    assert "shoreline" in info["variables"]
    assert info["time_range"][0].startswith("2010-01-01")


def test_peek_never_raises(tmp_path):
    bad = tmp_path / "broken.nc"
    bad.write_bytes(b"not a netcdf file")
    info = inv.peek(bad)
    assert info["kind"] == "error"


def test_build_inventory_outputs(fake_raw, tmp_path):
    out = tmp_path / "inv"
    files, teams = inv.build_inventory(fake_raw, out)
    for name in ("files.csv", "teams.csv", "peek.jsonl", "INVENTORY.md"):
        assert (out / name).exists()
    t = teams.set_index(["top", "team"])
    assert t.loc[("UserSubmissions", "TEAM_A"), "n_files"] == 2
    assert t.loc[("UserSubmissions", "TEAM_A"), "template_files_found"] == 1
    assert t.loc[("UserSubmissions", "TEAM_B"), "template_files_found"] == 0
    md = (out / "INVENTORY.md").read_text()
    assert "UserSubmissions/TEAM_A" in md and "Datetime" in md


def test_load_manifest_from_globus_json(tmp_path):
    # shape produced by `globus ls -r -F json` (names are relative paths)
    listing = {"DATA": [
        {"name": "CCOST", "type": "dir", "size": 0, "last_modified": "2026-01-01 00:00:00+00:00"},
        {"name": "BRIE", "type": "dir", "size": 0, "last_modified": None},
        {"name": "CCOST/pred.csv", "type": "file", "size": 2_000_000, "last_modified": None},
        {"name": "CCOST/sub/run.nc", "type": "file", "size": 5_000_000, "last_modified": None},
        {"name": "BRIE/out.csv", "type": "file", "size": 1_000, "last_modified": None},
        {"name": "notes.txt", "type": "file", "size": 10, "last_modified": None},
    ]}
    path = tmp_path / "UserSubmissions.json"
    path.write_text(json.dumps(listing))
    df = inv.load_manifest(path)
    files = df[df["type"] == "file"]
    summary = inv.summarize_files(files).set_index("team")
    assert summary.loc["CCOST", "n_files"] == 2
    assert summary.loc["CCOST", "total_mb"] == pytest.approx(7.0)
    assert summary.loc["", "n_files"] == 1  # loose file directly under UserSubmissions
    assert set(df.loc[df["type"] == "dir", "team"]) == {"CCOST", "BRIE"}


def test_peek_matlab_files(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "model_output.mat"
    savemat(path, {"shoreline": np.zeros((10, 3)), "time": np.arange(10.0)})
    info = inv.peek(path)
    assert info["kind"] == "mat"
    names = {name for name, _, _ in info["variables"]}
    assert {"shoreline", "time"} <= names
    h5py = pytest.importorskip("h5py")
    path73 = tmp_path / "v73.mat"  # MATLAB -v7.3 files are HDF5 containers
    with h5py.File(path73, "w") as f:
        f.create_dataset("X", data=np.ones((4, 5)))
    info73 = inv.peek(path73)
    assert info73["kind"] == "mat-v7.3" and info73["variables"][0][0] == "X"
