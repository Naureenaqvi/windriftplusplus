"""
data_generator.py — Synthetic Dataset Generation
=================================================
Generates all 17 univariate synthetic datasets C1–C17 exactly as described
used across WinDrift++ experiments.

Experiment 1 (C1–C5):  Similar distributions — no drift expected
  Y2019 and Y2020 use identical monthly parameters.
  Monthly µ and σ increase linearly from BASE values.

Experiment 2 (C6–C10): Dissimilar distributions — drift expected
  Y2019: same as experiment 1.
  Y2020: monthly parameters are REVERSED (Dec2019→Jan2020, etc.)

Experiment 3 (C11–C17): Statistical characteristic changes
  Y2019: Normal(µ=0, σ=1) for all 12 months.
  Y2020: altered distribution per dataset (skewness, shape, location, etc.)

Each dataset is saved as a CSV with:
  - 30 rows (observations per month)
  - 24 columns: M01…M12 = Y2019, M13…M24 = Y2020
"""


# **** Shaping up provided code with parameters and correct indentation ****
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_MU, BASE_SIGMA, MU_STEP, SIGMA_STEP,
    VARIANT_SCALE, C11_17_REF_MU, C11_17_REF_SIGMA,
    N_OBS, RANDOM_SEED, DATASET_DIR, N_MONTHS
)


# ************************************************************
# Helper: compute monthly (µ, σ) for months 1..12 given scale factors
# ************************************************************

def monthly_params(mu_scale: float = 1.0, sigma_scale: float = 1.0):
    """
    Returns list of (µ, σ) for 12 months.
    Month i (1-based): µ = (BASE_MU + (i-1)*MU_STEP) * mu_scale
                        σ = (BASE_SIGMA + (i-1)*SIGMA_STEP) * sigma_scale
    Range of µ: 5.00 → 18.75 (base), σ: 1.00 → 3.75 (base)
    """
    params = []
    for i in range(1, 13):
        mu    = (BASE_MU    + (i - 1) * MU_STEP)    * mu_scale
        sigma = (BASE_SIGMA + (i - 1) * SIGMA_STEP) * sigma_scale
        params.append((mu, sigma))
    return params


def sample_month(rng, mu: float, sigma: float, n: int = N_OBS) -> np.ndarray:
    """Draw n observations from Normal(µ, σ)."""
    return rng.normal(mu, sigma, n)


# ************************************************************
# Experiment 1 & 2: C1–C10
# ************************************************************

def generate_c1_c10(dataset_id: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    C1–C5:  Similar — Y2019 == Y2020 params (no drift)
    C6–C10: Dissimilar — Y2020 reverses Y2019 params (drift)
    """
    mu_scale, sigma_scale = VARIANT_SCALE[dataset_id]
    params_y2019 = monthly_params(mu_scale, sigma_scale)

    is_dissimilar = dataset_id in {"C6", "C7", "C8", "C9", "C10"}
    if is_dissimilar:
        # Reverse the 12-month sequence: Dec2019→Jan2020, Nov2019→Feb2020, …
        params_y2020 = list(reversed(params_y2019))
    else:
        # Identical to Y2019
        params_y2020 = params_y2019

    col_names = [f"M{i:02d}" for i in range(1, N_MONTHS + 1)]
    data = {}

    for i, (mu, sigma) in enumerate(params_y2019, start=1):
        data[f"M{i:02d}"] = np.round(sample_month(rng, mu, sigma), 4)

    for i, (mu, sigma) in enumerate(params_y2020, start=13):
        data[f"M{i:02d}"] = np.round(sample_month(rng, mu, sigma), 4)

    return pd.DataFrame(data)[col_names]


# ************************************************************
# Experiment 3: C11–C17
# ************************************************************

def _ref_months(rng):
    """Y2019 reference: Normal(µ=0, σ=1) for all 12 months."""
    cols = {}
    for i in range(1, 13):
        cols[f"M{i:02d}"] = np.round(
            rng.normal(C11_17_REF_MU, C11_17_REF_SIGMA, N_OBS), 4
        )
    return cols


def generate_c11(rng: np.random.Generator) -> pd.DataFrame:
    """C11 — Skewness change: Y2020 ~ Chi-squared(k=4) [Figure 9b]"""
    data = _ref_months(rng)
    for i in range(13, 25):
        data[f"M{i:02d}"] = np.round(rng.chisquare(df=4, size=N_OBS), 4)
    return pd.DataFrame(data)[[f"M{i:02d}" for i in range(1, 25)]]


def generate_c12(rng: np.random.Generator) -> pd.DataFrame:
    """C12 — Shape change: Y2020 ~ Normal(µ=0, σ=10) [Figure 9c]"""
    data = _ref_months(rng)
    for i in range(13, 25):
        data[f"M{i:02d}"] = np.round(rng.normal(0, 10, N_OBS), 4)
    return pd.DataFrame(data)[[f"M{i:02d}" for i in range(1, 25)]]


def generate_c13(rng: np.random.Generator) -> pd.DataFrame:
    """C13 — Location change: Y2020 ~ Normal(µ=3, σ=1) [Figure 9d]"""
    data = _ref_months(rng)
    for i in range(13, 25):
        data[f"M{i:02d}"] = np.round(rng.normal(3, 1, N_OBS), 4)
    return pd.DataFrame(data)[[f"M{i:02d}" for i in range(1, 25)]]


def generate_c14(rng: np.random.Generator) -> pd.DataFrame:
    """C14 — Light tail: Y2020 ~ Normal(µ=0, σ=0.5) [Figure 9e]"""
    data = _ref_months(rng)
    for i in range(13, 25):
        data[f"M{i:02d}"] = np.round(rng.normal(0, 0.5, N_OBS), 4)
    return pd.DataFrame(data)[[f"M{i:02d}" for i in range(1, 25)]]


def generate_c15(rng: np.random.Generator) -> pd.DataFrame:
    """C15 — Heavy tail: Y2020 ~ Normal(µ=0, σ=2) [Figure 9f]"""
    data = _ref_months(rng)
    for i in range(13, 25):
        data[f"M{i:02d}"] = np.round(rng.normal(0, 2, N_OBS), 4)
    return pd.DataFrame(data)[[f"M{i:02d}" for i in range(1, 25)]]


def generate_c16(rng: np.random.Generator) -> pd.DataFrame:
    """C16 — Outliers: Y2020 ~ Rayleigh(σ=10) [Figure 9g]"""
    data = _ref_months(rng)
    for i in range(13, 25):
        data[f"M{i:02d}"] = np.round(rng.rayleigh(scale=10, size=N_OBS), 4)
    return pd.DataFrame(data)[[f"M{i:02d}" for i in range(1, 25)]]


def generate_c17(rng: np.random.Generator) -> pd.DataFrame:
    """C17 — Ties (repetitive values): Y2020 ~ Laplace(µ=0, σ=1) discretised [Figure 9h]"""
    data = _ref_months(rng)
    for i in range(13, 25):
        # Laplace then rounded to 1dp to create ties
        raw = rng.laplace(loc=0, scale=1, size=N_OBS)
        data[f"M{i:02d}"] = np.round(raw, 1)
    return pd.DataFrame(data)[[f"M{i:02d}" for i in range(1, 25)]]


# ************************************************************
# Master generator
# ************************************************************

def generate_all_datasets(save: bool = True) -> dict:
    """
    Generates and optionally saves all 17 synthetic datasets.
    Returns dict: { 'C1': DataFrame, ..., 'C17': DataFrame }
    Also returns metadata dict for labelling.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(DATASET_DIR, exist_ok=True)

    generators_c1_10 = {
        "C1":  lambda: generate_c1_c10("C1",  rng),
        "C2":  lambda: generate_c1_c10("C2",  rng),
        "C3":  lambda: generate_c1_c10("C3",  rng),
        "C4":  lambda: generate_c1_c10("C4",  rng),
        "C5":  lambda: generate_c1_c10("C5",  rng),
        "C6":  lambda: generate_c1_c10("C6",  rng),
        "C7":  lambda: generate_c1_c10("C7",  rng),
        "C8":  lambda: generate_c1_c10("C8",  rng),
        "C9":  lambda: generate_c1_c10("C9",  rng),
        "C10": lambda: generate_c1_c10("C10", rng),
    }

    generators_c11_17 = {
        "C11": lambda: generate_c11(rng),
        "C12": lambda: generate_c12(rng),
        "C13": lambda: generate_c13(rng),
        "C14": lambda: generate_c14(rng),
        "C15": lambda: generate_c15(rng),
        "C16": lambda: generate_c16(rng),
        "C17": lambda: generate_c17(rng),
    }

    all_generators = {**generators_c1_10, **generators_c11_17}

    # Dataset metadata (Table II)
    metadata = {
        "C1":  {"label": "Base (similar)",         "group": "synthetic-similar"},
        "C2":  {"label": "Scale -10% (similar)",   "group": "synthetic-similar"},
        "C3":  {"label": "Scale -20% (similar)",   "group": "synthetic-similar"},
        "C4":  {"label": "Scale +10% (similar)",   "group": "synthetic-similar"},
        "C5":  {"label": "Scale +20% (similar)",   "group": "synthetic-similar"},
        "C6":  {"label": "Base (dissimilar)",       "group": "synthetic-dissimilar"},
        "C7":  {"label": "Mean -10% (dissimilar)", "group": "synthetic-dissimilar"},
        "C8":  {"label": "Mean -20% (dissimilar)", "group": "synthetic-dissimilar"},
        "C9":  {"label": "Mean +10% (dissimilar)", "group": "synthetic-dissimilar"},
        "C10": {"label": "Mean +20% (dissimilar)", "group": "synthetic-dissimilar"},
        "C11": {"label": "Skewness shift",         "group": "synthetic-characteristic"},
        "C12": {"label": "Shape change",           "group": "synthetic-characteristic"},
        "C13": {"label": "Location shift",         "group": "synthetic-characteristic"},
        "C14": {"label": "Variance decrease",      "group": "synthetic-characteristic"},
        "C15": {"label": "Variance increase",      "group": "synthetic-characteristic"},
        "C16": {"label": "Outlier distribution",   "group": "synthetic-characteristic"},
        "C17": {"label": "Discretised (ties)",     "group": "synthetic-characteristic"},
    }

    datasets = {}
    print("Generating synthetic datasets...")
    for cid, gen_fn in all_generators.items():
        df = gen_fn()
        datasets[cid] = df
        if save:
            path = os.path.join(DATASET_DIR, f"{cid}.csv")
            df.to_csv(path, index=False)
        print(f"  ✔ {cid:4s} — {metadata[cid]['label']:<30}  "
              f"Shape: {df.shape}  Drift: {metadata[cid]['drift']}")

    print(f"\n✔ All {len(datasets)} datasets ready → {DATASET_DIR}\n")
    return datasets, metadata


if __name__ == "__main__":
    datasets, meta = generate_all_datasets(save=True)
    # Quick sanity check
    for cid, df in datasets.items():
        y19_mean = df.iloc[:, :12].values.mean()
        y20_mean = df.iloc[:, 12:].values.mean()
        print(f"{cid}: Y2019 µ={y19_mean:6.2f}  Y2020 µ={y20_mean:6.2f}  "
              f"Δ={y20_mean - y19_mean:+.2f}")
