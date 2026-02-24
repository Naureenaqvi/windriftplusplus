# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.2.0] — 2026-02-25

### Changed
- `evaluation.py` rewritten to report what the detector found, with no external
  reference data required:
  - Removed `compute_confusion_matrix`, `compute_metrics`, `compute_time_metrics`
  - `summarise_dataset(dataset_id, results_df)` now returns `total_comparisons`,
    `flags_raised`, `flag_rate`, `first_flag_dbcount`, `MTD`, and per-test flag
    counts — purely observational output
- `DATASET_REGISTRY` in `config.py` stripped of all annotation fields; now holds
  only structural metadata needed at runtime (`label`, `group`, `multivariate`,
  `n_features`)
- `detection_summary.csv` replaces `metrics_summary.csv` — columns now reflect
  detection statistics rather than evaluation against a reference
- `main.py` (both modes): `REGENERATE_DATA` and `GENERATE_PLOTS` default to `False`
- All section divider comments standardised to `# **** section name ****` format
- Top-of-file comment added to every `.py` file

### Removed
- `GROUND_TRUTH` lookup removed from `config.py` entirely
- `actual_drift` and `drift_start` fields removed from `DATASET_REGISTRY`
- `GROUND_TRUTH` block removed from Mode 1 `main.py`
- `actual_drift` parameter removed from all `evaluation.py` function signatures

---

## [2.1.0] — 2026-01-14

### Added
- `windrift_pp_multivariate.py` — multivariate extension (C23-C30) applying
  per-feature WinDrift++ with feature-level majority aggregation
- `data_generator_multivariate.py` — generates all 8 multivariate datasets:
  - C23: Weather station (temperature, humidity, pressure)
  - C24: Energy system (demand, price, temperature)
  - C25: Ecological sensors (light, soil moisture, temperature, species count)
  - C26/C27: SEA concept — abrupt and gradual variants (3 features)
  - C28/C29: Agrawal concept — abrupt and gradual variants (salary, age, loan)
  - C30: Rotating hyperplane (5 features, gradual weight drift)
- Multivariate datasets stored as per-feature CSVs under `Datasets/<CID>/`
- `load_multivariate()` utility for reloading saved multivariate datasets
- `GROUPS`, `UNIVARIATE_IDS`, `MULTIVARIATE_IDS` constants for pipeline routing

### Changed
- `main.py` routes C23-C30 through `WinDriftPPMultivariate` automatically
- CI workflow updated: Mode 1 tests matrix Python 3.9-3.12; Mode 2 runs on 3.11
- `core_algorithm/`, `data_generation/`, `visualisation/` separated into distinct
  subfolders — data generation and visualisation can be excluded from deployment
  without affecting core algorithm or CI

### Fixed
- `plot_test_contributions()` called with stale extra argument — removed
- `plot_performance_comparison()` and `plot_voting_comparison()` wrong signatures fixed
- `summarise_dataset()` argument order corrected (`dataset_id` is first positional)

---

## [2.0.0] — 2026-01-14

### Added
- `data_generator_real.py` — five real-world-inspired univariate datasets (C18-C22):
  - C18: NOAA temperature — seasonal sinusoidal base + abrupt +4°C shift at Y2020
  - C19: Electricity demand — sinusoidal baseline + demand growth (mean +15, σ +5)
  - C20: Forest covertype elevation — Normal(2800m) → gradual drift to Normal(3100m)
  - C21: Insect population — Poisson(25) → Negative-Binomial (overdispersed)
  - C22: Air quality PM2.5 proxy — Gamma(2,15) → Gamma(2,30) scale doubling
- Separate random seed block for C18-C22 generators (`RANDOM_SEED + 100`)
- Voting comparison table extended to cover C1-C22

### Changed
- `main.py` restructured into Mode 1 (C1-C17) and Mode 2 (C1-C30) variants
- `requirements.txt` reorganised with inline comments separating runtime from
  dev/testing dependencies
- README updated with Mode 1 / Mode 2 structure, full dataset table, API examples

### Fixed
- Mode 1 `config.py` path resolution corrected for `core_algorithm/` subfolder layout

---

## [1.2.0] — 2025-11-30

### Added
- `DATASET_REGISTRY` dictionary in `config.py` centralising per-dataset attributes:
  label, group, multivariate flag, feature count
- Structured detection summary in run log grouped by dataset category

### Changed
- `main.py` inline dataset list replaced with import from `config.py`
- Drift heatmap x-axis labelled with dataset group headings

### Fixed
- `test_no_out_of_bounds_indices` split into two tests covering correct index
  ranges for Mode I and Mode II separately
- Kuiper test returned incorrect sign when all observations were identical —
  now correctly returns 0.0

---

## [1.1.1] — 2025-10-08

### Fixed
- EMD permutation test used unseeded RNG in multiprocessing context — now passes
  explicit `rng` instance throughout
- `voting_methods_comparison.csv` pivot had duplicate index entries when dataset
  IDs were unsorted — sort applied before pivot
- `plot_ecdf_comparisons()` raised `IndexError` on datasets with fewer than
  4 months of non-zero variance — guard clause added

### Changed
- All `np.random.RandomState` calls migrated to `np.random.default_rng()`
  (NumPy 1.17+ Generator API)
- Minimum scipy bumped to 1.10.0 following `anderson_ksamp` API change

---

## [1.1.0] — 2025-09-03

### Added
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`):
  matrix across Python 3.9-3.12; Codecov upload on Python 3.11
- Issue templates: `bug_report.md`, `feature_request.md`
- Pull request template with algorithmic change checklist
- `pytest.ini` suppressing `scipy.stats.anderson_ksamp` deprecation warnings
- `tests/__init__.py` making tests a proper importable package

### Changed
- `test_windrift_pp.py` reorganised into 7 named test classes
- CONTRIBUTING.md: PRs for algorithmic changes must reference the relevant
  function or formula; purely implementation changes exempted

---

## [1.0.1] — 2025-08-11

### Fixed
- Mode II window index out-of-bounds for `win_level=12` at `dbcount=24` — off-by-one corrected
- `visualisation.py` `plot_dataset_means()` divide-by-zero on datasets where
  Y2019 and Y2020 have identical means — guard added
- C17 Laplace rounding used `np.round(x, 4)` instead of `np.round(x, 1)` —
  corrected to produce the intended tie structure in Y2020

### Changed
- `config.py` moved from project root into `core_algorithm/` subfolder
- `.gitignore` updated to exclude `Results/plots/`, `Results/tables/per_dataset/`,
  and `Results/logs/` while retaining summary CSVs

---

## [1.0.0] — 2025-07-22

### Added
- Initial implementation of WinDrift++
- `ecdf.py` — ECDF computation with tie-handling via
  `scipy.stats.percentileofscore(kind='weak')`
- `statistical_tests.py` — five statistical distance tests:
  Kolmogorov-Smirnov, Kuiper, Cramér-von Mises, Anderson-Darling,
  Earth Mover's Distance (Wasserstein-1, 500-permutation critical value)
- `voting.py` — five voting methods: majority, plurality, Borda count,
  plurality with elimination, pairwise comparison
- `windrift_pp.py` — core detector with Mode I (Consecutive) and
  Mode II (Corresponding); window levels W = 1, 3, 6, 12 months
- `data_generator.py` — 17 synthetic datasets (C1-C17):
  - C1-C5: similar distributions, no change introduced
  - C6-C10: dissimilar distributions, Y2020 reverses Y2019 monthly order
  - C11-C17: statistical characteristic changes (skewness, shape, location,
    variance, outliers, ties)
- `evaluation.py` — detection statistics: flag counts, flag rates, MTD,
  per-test contributions
- `visualisation.py` — dataset overview, ECDF comparisons, distribution
  histograms, drift heatmaps, test contribution charts, voting comparison
- `main.py` — five-step pipeline: generate → detect → vote → summarise → visualise
- `config.py` — all algorithm parameters: `WIN_LEV_SIZE=[1,3,6,12]`,
  `MAX_CYC_LEN=12`, `ALPHA=0.05`, `N_OBS=30`, `RANDOM_SEED=112`
- `Datasets/` — 17 CSVs, 30 obs × 24 months each
- `Results/tables/voting_methods_comparison.csv`
