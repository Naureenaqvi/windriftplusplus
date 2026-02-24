"""
data_generator_real.py — Real-World Univariate Dataset Generation (C18–C22)
=============================================================================
Generates five univariate datasets inspired by real-world concept drift
benchmarks. Each replicates the statistical character of a known dataset
while remaining fully synthetic and reproducible via RANDOM_SEED.

Dataset structure (matches C1-C17):
  30 rows (observations)  x  24 columns (M01-M12 = Y2019, M13-M24 = Y2020)

C18 — NOAA Temperature
  Inspired by NOAA weather station data used in concept drift research.
  Y2019: monthly temps following realistic seasonal curve (cold→hot→cold).
  Y2020: abrupt mean shift (+4°C, mimicking El Niño / station relocation).
  Drift: TRUE, starts at M13 (dbcount=12).

C19 — Electricity Demand
  Inspired by the UCI Electricity dataset (Harries, 1999).
  Y2019: sinusoidal daily demand pattern with weekly variance.
  Y2020: sustained upward trend in both mean and variance (demand growth).
  Drift: TRUE, starts at M13 (dbcount=12).

C20 — Forest Covertype Elevation
  Inspired by the UCI Covertype dataset elevation feature (Blackard, 1998).
  Y2019: Normal(2800m, 200m) — typical forest elevation band.
  Y2020: Gradual drift starting M13, shifting mean to 3100m (tree-line).
  Drift: TRUE, starts at M14 (dbcount=13).

C21 — Insect Population Counts
  Inspired by count-based insect monitoring streams.
  Y2019: Poisson(lambda=25) — stable colony counts.
  Y2020: abrupt change to Negative-Binomial(overdispersed), variance spike.
  Drift: TRUE, starts at M14 (dbcount=13).

C22 — Air Quality (PM2.5 proxy)
  Inspired by urban air quality monitoring streams (heavy-tailed, seasonal).
  Y2019: Gamma(shape=2, scale=15) — typical clean-season distribution.
  Y2020: Gamma(shape=2, scale=30) — pollution event / scale shift.
  Drift: TRUE, starts at M13 (dbcount=12).
"""


# **** Shaping up provided code with parameters and correct indentation ****
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import N_OBS, N_MONTHS, RANDOM_SEED, DATASET_DIR


def _make_df(data: dict) -> pd.DataFrame:
    cols = [f"M{i:02d}" for i in range(1, N_MONTHS + 1)]
    return pd.DataFrame(data)[cols]


# ************************************************************
def generate_c18(rng: np.random.Generator) -> pd.DataFrame:
    """
    C18 — NOAA Temperature (°C)
    Y2019: seasonal sinusoidal curve  mu(m) = 10 + 12*sin(pi*(m-1)/11)
    Y2020: same curve + abrupt +4°C shift (El Nino / station move)
    """
    data = {}
    for i in range(1, 13):
        mu = 10 + 12 * np.sin(np.pi * (i - 1) / 11)
        data[f"M{i:02d}"] = np.round(rng.normal(mu, 2.5, N_OBS), 2)
    for i in range(13, 25):
        mu = 10 + 12 * np.sin(np.pi * (i - 13) / 11) + 4.0   # +4C shift
        data[f"M{i:02d}"] = np.round(rng.normal(mu, 2.5, N_OBS), 2)
    return _make_df(data)


def generate_c19(rng: np.random.Generator) -> pd.DataFrame:
    """
    C19 — Electricity Demand (kWh/day, normalised to 100)
    Y2019: sinusoidal seasonal pattern, sigma=8
    Y2020: upward drift — mean +15, sigma +5 (demand growth)
    """
    data = {}
    for i in range(1, 13):
        mu = 80 + 10 * np.cos(2 * np.pi * (i - 1) / 12)
        data[f"M{i:02d}"] = np.round(rng.normal(mu, 8, N_OBS), 2)
    for i in range(13, 25):
        mu = 95 + 10 * np.cos(2 * np.pi * (i - 13) / 12)     # +15 mean
        data[f"M{i:02d}"] = np.round(rng.normal(mu, 13, N_OBS), 2)  # +5 sigma
    return _make_df(data)


def generate_c20(rng: np.random.Generator) -> pd.DataFrame:
    """
    C20 — Covertype Elevation (metres)
    Y2019: Normal(2800, 200) — stable elevation band
    Y2020: gradual shift starting M13, reaching Normal(3100, 220) by M24
    """
    data = {}
    for i in range(1, 13):
        data[f"M{i:02d}"] = np.round(rng.normal(2800, 200, N_OBS), 1)
    for i in range(13, 25):
        frac = (i - 13) / 11               # 0.0 → 1.0
        mu   = 2800 + frac * 300           # 2800 → 3100
        sig  = 200  + frac * 20            # 200  → 220
        data[f"M{i:02d}"] = np.round(rng.normal(mu, sig, N_OBS), 1)
    return _make_df(data)


def generate_c21(rng: np.random.Generator) -> pd.DataFrame:
    """
    C21 — Insect Population Counts
    Y2019: Poisson(lambda=25) — stable colony
    Y2020: Negative-Binomial(n=5, p=0.17) — overdispersed, higher variance
           mean ≈ 24.7, var ≈ 145 (vs Poisson var=25)
    """
    data = {}
    for i in range(1, 13):
        data[f"M{i:02d}"] = rng.poisson(lam=25, size=N_OBS).astype(float)
    for i in range(13, 25):
        # NegBinomial: n=5, p=0.17 → mean=n*(1-p)/p ≈ 24.7, var ≈ 145
        data[f"M{i:02d}"] = rng.negative_binomial(n=5, p=0.17, size=N_OBS).astype(float)
    return _make_df(data)


def generate_c22(rng: np.random.Generator) -> pd.DataFrame:
    """
    C22 — Air Quality PM2.5 proxy (µg/m³)
    Y2019: Gamma(shape=2, scale=15) — clean-season, mean=30
    Y2020: Gamma(shape=2, scale=30) — pollution event, mean=60 (scale doubled)
    """
    data = {}
    for i in range(1, 13):
        data[f"M{i:02d}"] = np.round(rng.gamma(shape=2, scale=15, size=N_OBS), 2)
    for i in range(13, 25):
        data[f"M{i:02d}"] = np.round(rng.gamma(shape=2, scale=30, size=N_OBS), 2)
    return _make_df(data)


# ************************************************************
def generate_real_univariate(save: bool = True) -> dict:
    """
    Generate C18–C22. Returns dict {cid: DataFrame}.
    Saves each CSV to DATASET_DIR if save=True.
    """
    rng = np.random.default_rng(RANDOM_SEED + 100)   # separate seed block
    os.makedirs(DATASET_DIR, exist_ok=True)

    generators = {
        "C18": generate_c18,
        "C19": generate_c19,
        "C20": generate_c20,
        "C21": generate_c21,
        "C22": generate_c22,
    }

    datasets = {}
    print("Generating real-world univariate datasets (C18-C22)...")
    for cid, fn in generators.items():
        df = fn(rng)
        datasets[cid] = df
        if save:
            path = os.path.join(DATASET_DIR, f"{cid}.csv")
            df.to_csv(path, index=False)
        y19 = df.iloc[:, :12].values.mean()
        y20 = df.iloc[:, 12:].values.mean()
        print(f"  ✔ {cid}  Shape: {df.shape}  Y2019 µ={y19:.2f}  Y2020 µ={y20:.2f}  Δ={y20-y19:+.2f}")
    print(f"\n✔ C18-C22 ready → {DATASET_DIR}\n")
    return datasets


if __name__ == "__main__":
    generate_real_univariate(save=True)
