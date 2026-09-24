# Analysis plan (draft)

## Goal

Describe, for each ShoreShop3 model, at which time scales (and, for the
alongshore models, at which length scales) its shoreline positions agree with
the observations, and where they do not. The wavelet transform splits a series
by period and shows when each period is active, so model and observations can
be compared band by band and through time.

## Scope

**Models.** About ten single-profile models, each run for at least two
transects (the Duck profiles yFRF = 1 and 1006), and a smaller number of
alongshore models covering hundreds of transects (the `mipNC_*` templates list
about 970 North Carolina transects).

**Periods.** Where teams provide them:

| Run | Period | Observations | What the comparison can show |
|---|---|---|---|
| Hindcast | 1980-2023 | FRF surveys to 2019-12-06; CoastSat shorelines before 2020 | skill against observations, band by band |
| Century hindcast | 1871-2012 (template names) | historical shorelines only | model-to-model agreement; decadal periods that a 30-year record cannot resolve |
| Projections | 2024-2100 | none | how the variance at each scale changes, and whether the spread comes from the models, the GCMs (5), the scenarios (SSP2-4.5, SSP5-8.5) or the tropical-cyclone waves |

## Pipeline

| Step | What | Status |
|---|---|---|
| 0. Inventory | Mirror the Globus collection; list what each team submitted against `SubmissionTemplates/` | done (`outputs/inventory/`) |
| 1. Harmonise | Read each team's files into one table per site (model x transect x time), with the same units, sign convention and transect IDs | Duck: files read onto one grid (`src/shoreshop3/duck.py`); units, sign convention and shoreline definition not yet checked. NC to do |
| 2. Sampling | Put observations on a regular grid, flag long gaps, and read each model on the observation days so both series share the same sampling | done for Duck |
| 3. Time scales | Per transect: wavelet power of observations and models, coherence, and skill per band (`compare_models`) | Duck first pass done |
| 4. Length scales | The same tools along the coast (`dt` = transect spacing) for the alongshore models | to do |
| 5. Forcing and response | Coherence between forcing (wave energy, water level) and shoreline, for observations and for each model: does a model respond to the same forcing, at the same scales, with the same lag? | to do |
| 6. Synthesis | Per model: where it agrees with the observations and where it does not | after steps 2-5 are checked |

## Period bands used for Duck

| Band | Periods | Notes |
|---|---|---|
| Monthly | 45-120 days | Month-to-month changes that the surveys can see. Surveys are 13-40 days apart depending on the years, so this band is only partly resolved when they are sparse. Not single storms. |
| Sub-annual | 4-8 months | |
| Annual | 8-18 months | the seasonal cycle |
| Interannual | 1.5-4 years | |
| Multi-year | 4-8 years | 2.6-2.7 usable cycles over the whole record (1.3 over 1988-2019): exploratory, not ranked. |
| Trend | the whole window | linear slope, compared directly rather than with wavelets |

A band is a range of time scales, not a process. Links to processes (storm
recovery, seasonal exchange, climate modes, nourishment) are hypotheses to test
in step 5 and against intervention records such as
`InputData/NCnourishmentsUntil28-Oct-2025.csv`.

Alongshore bands (placeholder): below 0.5 km, 0.5-5 km, above 5 km.

## Metrics (per model and band)

* `std_ratio`: amplitude of the band signal, model / observed (below 1: too damped).
* `corr`, `rmse`, `nse`: agreement of the band-limited signals, over the times
  where the whole band is trustworthy (`valid_frac`, `valid_start` to `valid_end`, `n_cycles`).
* `mean_rsq`, `sig_frac`: wavelet coherence, i.e. co-variation regardless of
  amplitude, over every trustworthy cell (`valid_cell_frac`). `sig_frac` is a
  descriptive share of the band, not a band-level test, and its 95 % threshold
  is nominal (AR(1) surrogates on the regular grid).
* `phase_deg`, `lag`: timing where the series are coherent; positive = model lags.
* `var_frac_obs`: the band's share of the observed variance, counting only
  trustworthy cells.

## Status

* 2026-09-23: first pass for the Duck single-profile hindcasts (profiles 1 and
  1006, 33 runs from 10 teams, 1988-2019), `scripts/duck_wavelets.py`.
* 2026-09-24: variance shares count only trustworthy cells; the window is found
  automatically; constant models no longer stop the run; each band reports its
  support (dates, cycles); bands with fewer than three usable cycles are not
  ranked; family medians count each team once; figures in PNG and PDF.
* 2026-09-24: the main analysis uses the whole survey record (October 1980 to
  December 2019); runs that start later are left out of it and compared with
  the others over a common window (`--window common`). That window is chosen
  after setting aside runs with missing values inside it, so they do not
  shorten it. A constant model is masked with its own gap flags. Each figure
  shows the support of its own metric (% of time for NSE and amplitude, % of
  cells for coherence and timing), significance levels are called nominal,
  and each analysis records which runs took part and which
  were left out (with reasons), the grid, the code version and checksums of the
  input files (`models.csv`, `run_info.json`).

## Known limitations and next steps

1. **Resolution at short periods.** Test how well a signal of known period,
   amplitude and phase is recovered after sampling on the real survey dates and
   interpolating. Use the result as a time-varying lower period limit (a
   second mask, like the cone of influence), then fix the shortest band.
2. **Significance.** The red-noise surrogates are generated on the regular
   grid. They should go through the same sampling, interpolation and masking
   as the data, with the red-noise model fitted to the irregular surveys; then
   check with independent simulations that about 5 % of cells exceed the 95 %
   level.
3. **Coherence mask.** Spread the gap mask with the same smoothing that the
   coherence uses.
4. **Band-level statements.** Build the chance distribution of `sig_frac` for
   each band from the surrogates, and use block resampling to compare models.
5. **Phase.** Report how consistent the phase is within a band; give a lag only
   where it is consistent, otherwise by period of time.
6. **Sensitivity.** With and without the linear trend; with and without the
   June 2017 step at profile 1006.
7. **Submission checks.** Units, sign convention and shoreline definition for
   each team; the calibration and assimilation periods each team used.
8. **Further data.** The alongshore models (step 4), the century hindcast and
   the projections (model-to-model comparisons).

## Open questions

1. Which window for the main Duck analysis? The whole survey record (from
   October 1980) keeps 15 of the 33 runs (7 of the 10 teams). Starting in late
   June 1984 keeps 32 runs at profile 1 and 30 at profile 1006, from all 10
   teams, and shortens the record by about four years.
2. Are the 2020-2023 observations held back for a blind test? The FRF surveys
   in `InputData/` end on 2019-12-06 and the CoastSat file is labelled
   "Pre-2020".
3. Which periods did each team calibrate on, and which runs assimilate data?
4. Are models expected to include nourishments (a nourishment list is in `InputData/`)?
5. Which model groupings make sense (the families in `config/duck_models.csv` are provisional)?
6. Beyond shoreline position, are other variables (dune, berm, barrier width) to be compared?
