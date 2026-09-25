# ShoreShop3 wavelet analysis

Organise the ShoreShop3 model submissions and compare them with the
observations using wavelets: at which time scales (and, for the alongshore
models, which length scales) each model agrees with the observations, and
where it does not.

The plan, known limitations and open questions are in
[`docs/analysis_plan.md`](docs/analysis_plan.md).

## Data source

ShoreShop3 Globus guest collection *"Guest collection of /rt.attic.346.shoreshop3"*
(collection ID `5d24db1e-b934-4e35-9085-513be554aaa7`):

| Folder | Contents |
|---|---|
| `InputData/` | forcing and observations provided to modellers |
| `SubmissionTemplates/` | required submission format |
| `UserSubmissions/<TEAM>/` | one folder per model team (ANTOLINEZ, BRIE, CCOST, CEERD-CHL-..., ...) |
| `PublicSubmissions/` | public submissions |

> **Treat the collection as read-only.** Your Globus account may be able to
> create, rename or delete files there; do not. All work happens on the local
> copy in `data/raw/`. The other teams' submissions may not be public yet, so
> `data/` is excluded from git; keep the GitHub repository **private** until the
> organisers say otherwise.

## Layout

```
config/globus.env.example   collection ID, folders, destination (copy to globus.env)
scripts/
  globus_manifest.sh        list the collection + sizes per team (no download)
  globus_sync.sh            one-way mirror to data/raw (checksum-based, re-runnable)
  summarize_manifest.py     table of files and sizes per team from a listing
  local_inventory.py        per-team file inventory + peek inside CSV/NetCDF/MAT
  duck_wavelets.py          Duck single-profile models vs FRF surveys (figures + tables)
config/duck_models.csv      Duck submissions: team, model label, provisional family
src/shoreshop3/
  wavelets.py               CWT, cross-wavelet, coherence, significance, per-band skill
  duck.py                   read Duck submissions and FRF surveys onto one grid
  timeseries.py             irregular observations -> regular grid + gap flags
  inventory.py              inventory helpers used by the scripts
  plotting.py               wavelet power / coherence maps, spectra, skill heatmaps
  paths.py                  project paths (override with SHORESHOP3_DATA)
tests/                      pytest suite (synthetic signals with known answers)
data/                       raw/ interim/ processed/ manifests/  (git-ignored)
outputs/                    figures and tables (git-ignored)
```

## Setup

```bash
# conda / mamba
conda env create -f environment.yml
conda activate shoreshop3
pip install -e .

# or plain venv
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,globus,crosscheck]"

pytest            # ~15 s; checks the wavelet code against known answers and pycwt
```

## Get the data

1. `globus login` (opens a browser; use your Duke login).
2. Install and start [Globus Connect Personal](https://www.globus.org/globus-connect-personal)
   so your computer is a Globus endpoint. Keep this repo somewhere under your home
   folder but **not** in Desktop/Documents/iCloud/Dropbox (macOS privacy
   rules and cloud sync get in the way of large transfers), e.g. `~/Projects/`.
3. `cp config/globus.env.example config/globus.env` (defaults are fine for a laptop).
4. See how big it is before downloading:
   ```bash
   scripts/globus_manifest.sh        # writes data/manifests/<time>/summary.csv
   ```
5. Mirror it (only reads from the collection; re-run any time to pick up resubmissions):
   ```bash
   scripts/globus_sync.sh SubmissionTemplates InputData   # small folders first
   scripts/globus_sync.sh                                 # everything
   ```
   If the data are too big for the laptop, set `DEST_COLLECTION` and `DEST_ROOT`
   in `config/globus.env` to a Duke cluster collection and folder.
6. Inventory what each team submitted:
   ```bash
   python scripts/local_inventory.py   # -> outputs/inventory/INVENTORY.md, teams.csv, files.csv
   ```

## Duck single-profile hindcasts (first analysis)

Needs `InputData/hindcast_1980_2023/shorelines_and_profiles/FRF_Profiles.zip` and
the teams' `mipDuck_1980-2023*.csv` files in `data/raw/`.

```bash
python scripts/duck_wavelets.py             # whole survey record -> outputs/duck_1980-2023/
python scripts/duck_wavelets.py --window common --out outputs/duck_1980-2023_common
python scripts/duck_wavelets.py --profiles 1006 --end 2017-06-13 --no-maps \
    --out outputs/duck_1980-2023_1006_pre2017 --note "Profile 1006 before the June 2017 step"
```

* Surveys at FRF profiles yFRF = 1 and 1006 go on a weekly grid. Gaps longer
  than 60 days are masked.
* Window. By default (`--window surveys`) it is the whole survey record,
  October 1980 to December 2019; runs that do not cover it are left out, with
  the reason. `--window common` uses the longest window that the surveys and
  every run that can take part cover: a run with missing values inside the
  candidate window is set aside and the window is recomputed without it, so it
  does not shorten the window for the others. `--start` and `--end` fix either
  end by hand.
* Models are read on the survey days and interpolated the same way
  (`--sampling surveys`), so both series carry the same sampling filter.
* Model labels and provisional families: `config/duck_models.csv` (a new
  submission without a row still runs, labelled by its file name).
* Figures are written as PNG (300 dpi; the two atlases 220 dpi) and PDF
  (`--dpi`, `--no-pdf`); the one-per-model maps in `wtc/` are PNG only.
* `outputs/duck_1980-2023/README.md` lists the runs taken part and left out
  (with reasons), the results by band (with the support behind each score:
  % of time for NSE and amplitude, % of cells for coherence and timing, and
  the cycles in all and in the longest continuous stretch), family and team
  medians, trends, data checks (outliers, steps, duplicate or constant
  submissions) and caveats.
* `models.csv` lists every run per profile with its first and last day, whether
  it took part and why not. `run_info.json` records the settings, the window and
  what set it, the grid of each profile, the git commit, software versions and
  the SHA-256 of every input file.
* Coherence significance levels (nominal: AR(1) surrogates on the regular grid)
  are cached in `data/interim/wtc_sig/`: the first run takes about 5 minutes,
  later runs about 2 minutes (a new window length needs a new cache).

## Wavelet toolkit in 10 lines

```python
from shoreshop3 import wavelets as wv, regularize

reg = regularize(obs_time, obs_x, "1D", max_gap="30D")   # irregular obs -> daily grid
# model_a, model_b: daily model output on the same grid (reg.time)
bands = {"1-6 months": (30, 180), "annual": (180, 540), "interannual": (540, 2600)}
skill = wv.compare_models(reg.values, {"A": model_a, "B": model_b}, dt=reg.dt,
                          bands=bands, invalid_obs=reg.gap, n_surrogates=300)
coh = wv.wavelet_coherence(reg.values, model_a, reg.dt, invalid_x=reg.gap)
sig = wv.coherence_significance(len(reg.values), reg.dt, wv.ar1(reg.values), wv.ar1(model_a))
```

`skill` has one row per model and band. Columns: the support of each score
(share of the time steps for the signal scores, share of the time-period cells
for coherence, first and last usable time, number of cycles in all and in the
longest continuous stretch), the band's share of the observed variance over the
same times as the signal scores, amplitude ratio, correlation, RMSE, NSE of the
band-limited signals, mean coherence, share of the band with coherence above
the 95 % level, and phase/lag (positive = model lags observations). A model
without variability gets amplitude scores and a note instead of stopping the
run.

Conventions and caveats:

* Morlet wavelet (omega0 = 6) following Torrence & Compo (1998); coherence
  smoothing and Monte Carlo significance follow Grinsted et al. (2004).
  Validated against `pycwt` in `tests/test_wavelets.py`.
* `dt` can be days, years or metres (alongshore transforms); periods come back
  in the same unit.
* Results inside the cone of influence (record edges), or where more than 25 %
  of a wavelet's energy falls on long filled gaps, are left out of every
  statistic (so a short gap only removes short periods). A band's variance
  share compares its band signal with the series over the same times. The
  coherence mask does not yet include the extra spread from the coherence
  smoothing.
* Where a series has essentially no variance at some period (e.g. a smooth
  model at short periods), coherence is set towards 0 rather than the unstable 0/0.
* The AR1 red-noise test is a reference background, not proof of a process;
  shoreline series are very persistent (AR1 close to 1). For interpolated
  surveys the surrogates do not yet reproduce the sampling, and the share of
  significant cells is descriptive, not a band-level test (see
  `docs/analysis_plan.md`).

## Start the git repository

```bash
cd ~/Projects/shoreshop3-wavelets
git init -b main
git add .
git status            # check: nothing under data/ or outputs/ is listed
git commit -m "Project scaffold: Globus mirror, inventory and wavelet toolkit"
# private GitHub repo with the GitHub CLI:
gh repo create shoreshop3-wavelets --private --source=. --push
```

## References

* Torrence, C. & Compo, G. P. (1998). A practical guide to wavelet analysis. *BAMS* 79, 61-78.
* Torrence, C. & Webster, P. J. (1999). Interdecadal changes in the ENSO-monsoon system. *J. Climate* 12, 2679-2690.
* Grinsted, A., Moore, J. C. & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. *Nonlin. Processes Geophys.* 11, 561-566.
