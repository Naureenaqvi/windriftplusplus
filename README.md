# WinDrift++

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: IEEE TAI 2023](https://img.shields.io/badge/Paper-IEEE%20TAI%202023-red.svg)](https://doi.org/10.1109/TAI.2023.3293657)

Python implementation of **WinDrift++** — a diversified detector for concept drift.

> N. Naqvi, S. u. Rehman, and M. Z. Islam, “Windrift++:a diversified detector for concept drift,” IEEE Transactions on Artificial Intelligence, pp. 1–16, 2025.

---

## Algorithm Overview

WinDrift++ compares data distributions across time windows using five statistical tests:

| Test | Statistic | Detects |
|---|---|---|
| Kolmogorov–Smirnov (KS) | Max CDF difference | Location shifts |
| Kuiper | Combined max +/− deviation | Asymmetric shifts |
| Cramér–von Mises (CVM) | Summed squared differences | Overall shape |
| Anderson–Darling (AD) | Tail-weighted CDF distance | Tail changes |
| Earth Mover's Distance (EMD) | Optimal transport cost | Any distributional change |

Results are combined via **majority vote** (≥3/5 tests agree → drift flagged).

**Two detection modes:**
- **Mode I (Consecutive):** compares adjacent windows within the same year — detects drift onset
- **Mode II (Corresponding):** compares same-period windows across years — confirms sustained drift

**Window levels:** monthly (W=1), quarterly (W=3), semi-annual (W=6), annual (W=12)

---

## Project Structure

```
WinDriftPP_Mode2/              ← Full project root (use Mode1 for C1-C17 only)
├── main.py                    ← Entry point — run this
├── core_algorithm/
│   ├── config.py              ← All algorithm parameters and dataset definitions
│   ├── ecdf.py                ← Empirical CDF computation
│   ├── statistical_tests.py   ← KS, Kuiper, CVM, AD, EMD tests
│   ├── voting.py              ← 5 voting methods
│   ├── windrift_pp.py         ← Core detector: Mode I & II (univariate)
│   ├── windrift_pp_multivariate.py  ← Multivariate extension (C23-C30)
│   └── evaluation.py          ← Detection statistics and flag rates
├── data_generation/           ← Standalone — can exclude from deployment
│   ├── data_generator.py      ← C1-C17 synthetic univariate
│   ├── data_generator_real.py ← C18-C22 real-world univariate
│   └── data_generator_multivariate.py  ← C23-C30 multivariate
├── visualisation/             ← Standalone — can exclude from deployment
│   └── visualisation.py       ← Charts: overview, ECDF, heatmaps, ablation
├── tests/
│   └── test_windrift_pp.py    ← 47 unit tests
├── Datasets/                  ← Generated on first run (or pre-loaded)
└── Results/                   ← Output tables, plots, logs
```

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/YOUR-USERNAME/windrift-pp.git
cd windrift-pp
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline (generates data, runs detection, saves results)
python main.py
```

**Expected runtime:** ~90 seconds for the full C1-C30 pipeline.

To run C1-C17 only (Mode 1):
```bash
# Use the Mode 1 package — faster (~30s)
python main.py   # from WinDriftPP_Mode1/
```

---

## Configuration

Edit `core_algorithm/config.py` to change parameters:

```python
WIN_LEV_SIZE  = [1, 3, 6, 12]   # window sizes: monthly/quarterly/semi-annual/annual
MAX_CYC_LEN   = 12               # months per cycle
ALPHA         = 0.05             # significance level
N_OBS         = 30               # observations per monthly column
RANDOM_SEED   = 112
```

Toggle pipeline stages in `main.py`:
```python
REGENERATE_DATA = False   # True = regenerate CSVs from scratch
GENERATE_PLOTS  = False   # True = produce all charts after detection
```

---

## Datasets

### Mode 1 — Synthetic Univariate (C1-C17)

| Group | Datasets | Experiment | Description | Drift |
|---|---|---|---|---|
| Exp 1 | C1-C5 | Similar distributions | Y2019 ≈ Y2020 (±10–20% scale) | No |
| Exp 2 | C6-C10 | Dissimilar distributions | Y2020 reverses Y2019 monthly order | Yes |
| Exp 3 | C11-C13 | Significant statistical change | Skewness, shape, location shifts | Yes |
| Exp 3 | C14-C16 | Detectable characteristic change | Tail width, outlier distribution | Yes |
| Exp 3 | C17 | Ties (repetitive values) | Laplace discretised | No |

### Mode 2 — Extended (C18-C30)

| Group | Datasets | Experiment | Description | Drift |
|---|---|---|---|---|
| Exp 4 | C18-C22 | Real-world univariate | NOAA temp, electricity, covertype, insects, air quality | Yes |
| Exp 5 | C23-C25 | Real-world multivariate | Weather station, energy system, ecological sensors | Yes |
| Exp 6 | C26-C27 | SEA concept (abrupt/gradual) | 3 features, boundary shift | Yes |
| Exp 6 | C28-C29 | Agrawal concept (abrupt/gradual) | salary, age, loan distributions | Yes |
| Exp 6 | C30 | Rotating hyperplane | 5 features, gradual rotation | Yes |

Each dataset: **30 observations × 24 monthly columns** (M01-M12 = Y2019, M13-M24 = Y2020).  
Multivariate datasets: one CSV per feature, stored in `Datasets/<CID>/`.

---

## API Usage

### Univariate detection

```python
import sys, pandas as pd
sys.path.insert(0, 'core_algorithm')
from windrift_pp import WinDriftPP

df = pd.read_csv('Datasets/C6.csv')   # 30 rows × 24 columns
detector = WinDriftPP(win_lev_size=[1,3,6,12], max_cyc_len=12, alpha=0.05)
detector.detect(df, dataset_id='C6')
results = detector.to_dataframe()

# Mode II (Corresponding) drift detections
mode2 = results[results['mode'] == 2]
print(f"Drift flags: {(mode2['majority_flag']==1).sum()}/19")
```

### Multivariate detection

```python
from windrift_pp_multivariate import WinDriftPPMultivariate

# features = {feature_name: DataFrame} — one DataFrame per feature
features = {
    'temperature': pd.read_csv('Datasets/C23/temperature.csv'),
    'humidity':    pd.read_csv('Datasets/C23/humidity.csv'),
    'pressure':    pd.read_csv('Datasets/C23/pressure.csv'),
}
detector = WinDriftPPMultivariate()
results  = detector.detect(features, dataset_id='C23')
```

### Evaluation

```python
from evaluation import summarise_dataset
summary = summarise_dataset('C6', results)
# Returns: total_comparisons, flags_raised, flag_rate, first_flag_dbcount, MTD,
#          and per-test flag counts (ks_flags, kuiper_flags, cvm_flags, ad_flags, emd_flags)
```

---

## Results (Mode II, Majority Vote)

| Group | Drift injected | Detection Rate |
|---|---|---|
| C1-C5 Similar | No drift | 0-2/19 (≤10% FPR) |
| C6-C10 Dissimilar | Drift | 16-17/19 (84-89% recall) |
| C11-C13 Significant change | Drift | 19/19 (100%) |
| C14-C16 Detectable change | Drift | 13-19/19 |
| C17 Ties | No drift | 4/19 (21% FPR, AD-driven) |
| C18-C22 Real-world univariate | Drift | 13-19/19 |
| C23-C25 Real-world multivariate | Drift | 14-19/19 |
| C26-C30 Synthetic multivariate | Drift | 13-19/19 |

---

## Testing

```bash
cd WinDriftPP_Mode1   # or Mode2
python -m pytest tests/ -v
# Expected: 47 passed
```

---

## Citation

```bibtex
@ARTICLE{naqvi2023windriftpp,
  author={Naqvi, Naureen and Rehman, Sabih ur and Islam, Md Zahidul},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={WinDrift++:A diversified detector for concept drift}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Detectors;Concept drift;Time series analysis;Incremental learning;Robustness;Accuracy;Synthetic data;Mathematical models;Ensemble learning;Computational efficiency;Concept Drift Detection;Incremental Learning;Ensemble Methods;Statistical Hypothesis Testing;Data Streams;Non-Stationary Environments},
  doi={10.1109/TAI.2025.3619461}}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).  
This license covers the implementation only. The algorithm is the intellectual property of the original authors — please cite the paper in any research use.
