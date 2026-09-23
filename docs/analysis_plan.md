# Analysis plan (draft for discussion with Brad)

**Goal.** For each ShoreShop3 model, find the processes and the time and space
scales where it reproduces the observations, and where it does not. Wavelets
split each series by period and localise it in time, so we can say, e.g.,
"model X gets the seasonal cycle right but its storm response is too weak, and
it drifts after 2015".

## Pipeline

| Step | What | Output |
|---|---|---|
| 0. Inventory | Mirror the Globus collection; list what each team submitted (files, variables, transects, time span, format) against `SubmissionTemplates/` | `outputs/inventory/INVENTORY.md` |
| 1. Harmonise | Read every team's files into one schema: `model x transect x time` (plus ensemble member if any), same units, datum and transect IDs; observations and forcing on the same axes | `data/processed/*.nc` |
| 2. Regularise | Put observations on the model time step; flag long gaps (`regularize(..., max_gap=)`) so coefficients dominated by gaps are masked | gap flags |
| 3. Time-scale analysis | Per transect: CWT of obs and models, wavelet coherence obs-model with Monte Carlo significance, per-band skill (`compare_models`) | tidy table model x transect x band x metric |
| 4. Space-scale analysis | Same tools along the coast (`dt` = transect spacing): at each time, or on time-averaged/band-passed fields, to see which alongshore length scales each model captures | model x alongshore band x metric |
| 5. Process fingerprints | Coherence between forcing (wave energy / power, water level) and shoreline response, for observations and for each model: do models respond to the right forcing at the right scales, with the right lag? | coherence and lag per band |
| 6. Synthesis | Heatmaps (model x band) for amplitude ratio, NSE, coherence, lag; short model-by-model strengths/weaknesses table | figures for the paper |

## Period bands (to agree on)

Placeholders, to be tuned to the site, the record length and the sampling:

| Band | Periods | Processes (typical interpretation) |
|---|---|---|
| Event | 2 days - 1 month | storm erosion and early recovery |
| Sub-seasonal | 1 - 6 months | storm clustering, recovery |
| Seasonal | 6 - 18 months | seasonal cross-shore exchange, seasonal rotation |
| Interannual | 1.5 - 7 years | climate modes (ENSO/NAO), multi-year rotation |
| Long-term | > 7 years | trends: sea-level rise, sediment budget, nourishment |

Long bands are only evaluable away from the record ends (cone of influence);
`valid_frac` in the skill table says how much of the record each band uses.
The linear trend itself is better compared directly (slopes) than by wavelets.

Alongshore bands (placeholder): < 0.5 km (local), 0.5 - 5 km, > 5 km (embayment / regional).

## Metrics (per model and band)

* `std_ratio`: amplitude of the band signal, model / obs (too damped < 1 < too active)
* `corr`, `nse`, `rmse`: skill of the band-limited signals
* `mean_rsq`, `sig_frac`: wavelet coherence (timing and co-variation, independent of amplitude)
* `phase_deg`, `lag`: timing error; positive = model lags observations
* `var_frac_obs`: how much of the observed variance the band carries (weights the importance of a band)

## Status

* 2026-09: first pass for the Duck single-profile hindcasts (profiles 1 and 1006,
  33 submissions from 10 teams, 1988-2019): `scripts/duck_wavelets.py`, results in
  `outputs/duck_1980-2023/`. Bands used there: 1.5-4 months, 4-8 months, 8-18
  months, 1.5-4 years, 4-8 years (the surveys every ~2-6 weeks do not resolve
  shorter periods), plus the linear trend.

## Open questions

1. Evaluation data: the FRF surveys in `InputData/` end on 2019-12-06 (and the CoastSat file is labelled "Pre-2020"), so 2020-2023 looks like the blind test period. Everything up to 2019 is in-sample for the teams. Confirm with the organisers.
2. Which target variables: shoreline position only, or also dune, berm, barrier width / overwash?
3. Do all teams provide the same transects, time span and time step? Ensembles or multiple runs per team?
4. Observation sampling (surveys vs satellite): sets the shortest resolvable period.
5. Which teams / model types to group (e.g. equilibrium, one-line, process-based, data-driven, barrier models)?

## Caveats

* Coefficients near the record ends (cone of influence) and those dominated by long gaps are unreliable and are masked.
* AR1 red noise is the null hypothesis for significance; shoreline records are very persistent, so significance is a guide, not proof.
* Many transects x bands x models means many tests: report patterns, not single significant cells.
* Band reconstruction by inverse CWT is approximate (a few % of variance at the shortest periods).
