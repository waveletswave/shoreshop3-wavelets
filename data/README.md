# data/

Everything in here except this file and the `.gitkeep` placeholders is ignored by git.

| Folder | Contents | Written by |
|---|---|---|
| `raw/` | Read-only mirror of the ShoreShop3 Globus collection (`InputData/`, `SubmissionTemplates/`, `PublicSubmissions/`, `UserSubmissions/<team>/`). Never edit files here; re-run the sync instead. | `scripts/globus_sync.sh` |
| `manifests/` | Remote listings (`globus ls -r`) with file sizes, one folder per run | `scripts/globus_manifest.sh` |
| `interim/` | Caches and intermediate products (e.g. coherence significance levels) | analysis code |
| `processed/` | Harmonised datasets (model x transect x time), rebuilt from `raw/` by code | analysis code |

To keep the data elsewhere (external drive, cluster), set `SHORESHOP3_DATA=/path/to/data`
before running Python; `shoreshop3.paths` will point there.
