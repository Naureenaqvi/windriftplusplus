"""
data_generator_multivariate.py — Multivariate Dataset Generation (C23–C30)
===========================================================================
Generates 8 multivariate datasets (C23-C25 real-inspired, C26-C30 synthetic).

Storage format: each dataset is a FOLDER under Datasets/<CID>/ containing one
CSV per feature:
    Datasets/C23/temperature.csv      (30 obs × 24 months)
    Datasets/C23/humidity.csv
    Datasets/C23/pressure.csv

Each feature CSV uses the same 30-row × 24-column format as C1-C22.
WinDrift++ is applied independently to each feature; final drift decision
uses a per-dataset majority vote across features (see windrift_pp_multivariate.py).

──────────────────────────────────────────────────────────────────────────────
C23 — Weather Station (3 features: temperature, humidity, pressure)
  Y2019: seasonal sinusoidal patterns; correlated realistic values.
  Y2020: temperature shifts up +3°C; humidity shifts +8%; pressure unchanged.
  Drift: TRUE — 2 of 3 features drift.

C24 — Energy System (3 features: demand_kwh, price_cents, temperature)
  Y2019: demand and price anti-correlated with temperature (summer peak).
  Y2020: demand+price both shift up (energy crisis / demand growth).
  Drift: TRUE — 2 of 3 features drift.

C25 — Ecological Sensors (4 features: light, soil_moisture, temperature, count)
  Y2019: stable ecosystem readings.
  Y2020: gradual shift in all 4 features (habitat change).
  Drift: TRUE — all 4 features drift gradually.

C26 — SEAa (SEA abrupt, 3 features)
  Based on Street & Kim (2001) SEA concept. Features f1,f2,f3 ~ Uniform[0,10].
  Y2019: class boundary f1+f2 ≤ θ₁=8.
  Y2020: abrupt boundary change to θ₂=9 (abrupt drift at M13).
  Drift: TRUE, abrupt.

C27 — SEAg (SEA gradual, 3 features)
  Same as SEAa but boundary drifts gradually from θ=8 to θ=9 (M09→M16).
  Drift: TRUE, gradual.

C28 — AGRa (Agrawal abrupt, 3 features)
  Based on Agrawal et al. (1993). Features: salary, age, loan.
  Y2019: Normal(50k, 10k), Normal(35, 8), Gamma(2, 15k).
  Y2020: abrupt mean shift in salary (+15k) and age (+5).
  Drift: TRUE, abrupt.

C29 — AGRg (Agrawal gradual, 3 features)
  Same as AGRa but drift is gradual (linear interpolation M09→M16).
  Drift: TRUE, gradual.

C30 — Hyperplane (5 features: f1-f5 ~ Normal(0,1), rotating boundary)
  Y2019: hyperplane w·x = 0, w fixed.
  Y2020: w rotates gradually (concept drift via weight change).
  Drift: TRUE, gradual hyperplane rotation.
"""


# **** Shaping up provided code with parameters and correct indentation ****
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import N_OBS, N_MONTHS, RANDOM_SEED, DATASET_DIR


def _feature_df(data: dict) -> pd.DataFrame:
    cols = [f"M{i:02d}" for i in range(1, N_MONTHS + 1)]
    return pd.DataFrame(data)[cols]


def _save_multivariate(cid: str, features: dict, base_dir: str):
    """Save each feature as a separate CSV in Datasets/<CID>/"""
    out_dir = os.path.join(base_dir, cid)
    os.makedirs(out_dir, exist_ok=True)
    for fname, df in features.items():
        df.to_csv(os.path.join(out_dir, f"{fname}.csv"), index=False)


def _sinusoid(month_1based, base, amplitude, phase_offset=0):
    return base + amplitude * np.sin(np.pi * (month_1based - 1 + phase_offset) / 11)


# ************************************************************
# C23 — Weather Station
# ************************************************************
def generate_c23(rng):
    temp_data, hum_data, pres_data = {}, {}, {}
    for i in range(1, 13):
        temp_data[f"M{i:02d}"]  = np.round(rng.normal(10 + 12*np.sin(np.pi*(i-1)/11), 2.0, N_OBS), 2)
        hum_data[f"M{i:02d}"]   = np.round(rng.normal(60 -  8*np.sin(np.pi*(i-1)/11), 5.0, N_OBS), 1).clip(20, 100)
        pres_data[f"M{i:02d}"]  = np.round(rng.normal(1013, 3.0, N_OBS), 1)
    for i in range(13, 25):
        temp_data[f"M{i:02d}"]  = np.round(rng.normal(13 + 12*np.sin(np.pi*(i-13)/11), 2.0, N_OBS), 2)   # +3°C
        hum_data[f"M{i:02d}"]   = np.round(rng.normal(68 -  8*np.sin(np.pi*(i-13)/11), 5.0, N_OBS), 1).clip(20, 100)  # +8%
        pres_data[f"M{i:02d}"]  = np.round(rng.normal(1013, 3.0, N_OBS), 1)                               # no drift
    return {"temperature": _feature_df(temp_data),
            "humidity":    _feature_df(hum_data),
            "pressure":    _feature_df(pres_data)}


# ************************************************************
# C24 — Energy System
# ************************************************************
def generate_c24(rng):
    dem_data, price_data, temp_data = {}, {}, {}
    for i in range(1, 13):
        t = 15 + 10 * np.cos(2*np.pi*(i-1)/12)
        dem_data[f"M{i:02d}"]   = np.round(rng.normal(80 - 0.5*t, 8, N_OBS), 2)
        price_data[f"M{i:02d}"] = np.round(rng.normal(22 - 0.2*t, 3, N_OBS), 2)
        temp_data[f"M{i:02d}"]  = np.round(rng.normal(t, 2, N_OBS), 2)
    for i in range(13, 25):
        t = 15 + 10 * np.cos(2*np.pi*(i-13)/12)
        dem_data[f"M{i:02d}"]   = np.round(rng.normal(95 - 0.5*t, 10, N_OBS), 2)  # +15 demand
        price_data[f"M{i:02d}"] = np.round(rng.normal(30 - 0.2*t, 4,  N_OBS), 2)  # +8 price
        temp_data[f"M{i:02d}"]  = np.round(rng.normal(t, 2, N_OBS), 2)             # no drift
    return {"demand_kwh": _feature_df(dem_data),
            "price_cents": _feature_df(price_data),
            "temperature": _feature_df(temp_data)}


# ************************************************************
# C25 — Ecological Sensors (gradual drift all 4 features)
# ************************************************************
def generate_c25(rng):
    light_data, moist_data, temp_data, count_data = {}, {}, {}, {}
    for i in range(1, 13):
        light_data[f"M{i:02d}"]  = np.round(rng.normal(500, 60, N_OBS), 1)
        moist_data[f"M{i:02d}"]  = np.round(rng.normal(40, 8, N_OBS), 1).clip(5, 95)
        temp_data[f"M{i:02d}"]   = np.round(rng.normal(18, 3, N_OBS), 2)
        count_data[f"M{i:02d}"]  = rng.poisson(lam=30, size=N_OBS).astype(float)
    for i in range(13, 25):
        frac = (i - 13) / 11
        light_data[f"M{i:02d}"]  = np.round(rng.normal(500 + frac*100, 70, N_OBS), 1)    # +100 lux
        moist_data[f"M{i:02d}"]  = np.round(rng.normal(40  + frac*15,  8,  N_OBS), 1).clip(5, 95)  # +15%
        temp_data[f"M{i:02d}"]   = np.round(rng.normal(18  + frac*4,   3,  N_OBS), 2)    # +4°C
        lam = 30 + frac * 20
        count_data[f"M{i:02d}"]  = rng.poisson(lam=lam, size=N_OBS).astype(float)         # +20 count
    return {"light_lux":    _feature_df(light_data),
            "soil_moisture": _feature_df(moist_data),
            "temperature":  _feature_df(temp_data),
            "species_count": _feature_df(count_data)}


# ************************************************************
# C26 — SEAa: SEA concept, abrupt drift
# ************************************************************
def generate_c26(rng):
    """
    SEAa (SEA abrupt, covariate shift encoding).
    Features f1,f2 experience abrupt mean/variance shift at M13 (theta change);
    f3 is irrelevant (no drift) — matching SEA concept structure.
    Y2019: f1~N(4.5,1.5), f2~N(4.5,1.5), f3~Uniform[0,10]
    Y2020: f1~N(6.5,1.5), f2~N(5.5,1.5), f3~Uniform[0,10]  (abrupt +2 shift)
    """
    f1_d, f2_d, f3_d = {}, {}, {}
    for i in range(1, 13):
        f1_d[f"M{i:02d}"] = np.round(rng.normal(4.5, 1.5, N_OBS).clip(0, 10), 3)
        f2_d[f"M{i:02d}"] = np.round(rng.normal(4.5, 1.5, N_OBS).clip(0, 10), 3)
        f3_d[f"M{i:02d}"] = np.round(rng.uniform(0, 10, N_OBS), 3)
    for i in range(13, 25):   # abrupt shift
        f1_d[f"M{i:02d}"] = np.round(rng.normal(6.5, 1.5, N_OBS).clip(0, 10), 3)
        f2_d[f"M{i:02d}"] = np.round(rng.normal(5.5, 1.5, N_OBS).clip(0, 10), 3)
        f3_d[f"M{i:02d}"] = np.round(rng.uniform(0, 10, N_OBS), 3)
    return {"f1": _feature_df(f1_d), "f2": _feature_df(f2_d), "f3": _feature_df(f3_d)}


# ************************************************************
# C27 — SEAg: SEA concept, gradual drift
# ************************************************************
def generate_c27(rng):
    """
    SEAg (SEA gradual, covariate shift).
    Same structure as C26 but drift is gradual M09→M16 (means drift linearly).
    """
    f1_d, f2_d, f3_d = {}, {}, {}
    for i in range(1, 25):
        if i <= 8:     frac = 0.0
        elif i <= 16:  frac = (i - 8) / 8.0
        else:          frac = 1.0
        mu1 = 4.5 + frac * 2.0   # 4.5 → 6.5
        mu2 = 4.5 + frac * 1.0   # 4.5 → 5.5
        f1_d[f"M{i:02d}"] = np.round(rng.normal(mu1, 1.5, N_OBS).clip(0, 10), 3)
        f2_d[f"M{i:02d}"] = np.round(rng.normal(mu2, 1.5, N_OBS).clip(0, 10), 3)
        f3_d[f"M{i:02d}"] = np.round(rng.uniform(0, 10, N_OBS), 3)
    return {"f1": _feature_df(f1_d), "f2": _feature_df(f2_d), "f3": _feature_df(f3_d)}


# ************************************************************
# C28 — AGRa: Agrawal abrupt drift
# ************************************************************
def generate_c28(rng):
    """Agrawal abrupt: salary, age, loan. Abrupt mean shifts at M13."""
    sal_d, age_d, loan_d = {}, {}, {}
    for i in range(1, 13):
        sal_d[f"M{i:02d}"]  = np.round(rng.normal(50000, 10000, N_OBS), 0)
        age_d[f"M{i:02d}"]  = np.round(rng.normal(35, 8, N_OBS).clip(18, 70), 1)
        loan_d[f"M{i:02d}"] = np.round(rng.gamma(shape=2, scale=15000, size=N_OBS), 0)
    for i in range(13, 25):  # abrupt: salary +15k, age +5
        sal_d[f"M{i:02d}"]  = np.round(rng.normal(65000, 10000, N_OBS), 0)
        age_d[f"M{i:02d}"]  = np.round(rng.normal(40, 8, N_OBS).clip(18, 75), 1)
        loan_d[f"M{i:02d}"] = np.round(rng.gamma(shape=2, scale=15000, size=N_OBS), 0)
    return {"salary": _feature_df(sal_d), "age": _feature_df(age_d),
            "loan":   _feature_df(loan_d)}


# ************************************************************
# C29 — AGRg: Agrawal gradual drift
# ************************************************************
def generate_c29(rng):
    """Agrawal gradual: salary & age drift linearly M09→M16."""
    sal_d, age_d, loan_d = {}, {}, {}
    for i in range(1, 25):
        if i <= 8:       frac = 0.0
        elif i <= 16:    frac = (i - 8) / 8.0
        else:            frac = 1.0
        mu_sal = 50000 + frac * 15000
        mu_age = 35    + frac * 5
        sal_d[f"M{i:02d}"]  = np.round(rng.normal(mu_sal, 10000, N_OBS), 0)
        age_d[f"M{i:02d}"]  = np.round(rng.normal(mu_age, 8, N_OBS).clip(18, 75), 1)
        loan_d[f"M{i:02d}"] = np.round(rng.gamma(shape=2, scale=15000, size=N_OBS), 0)
    return {"salary": _feature_df(sal_d), "age": _feature_df(age_d),
            "loan":   _feature_df(loan_d)}


# ************************************************************
# C30 — Hyperplane (5 features)
# ************************************************************
def generate_c30(rng):
    """
    Hyperplane (5 features): abrupt distributional shift at M13.
    Y2019: f1-f5 ~ N(mu_y19, 1); mu_y19 = [0, 0, 0, 0, 0]
    Y2020: f1,f2 shift to N(3,1); f3 shifts to N(-2,1); f4,f5 stable.
    Models a hyperplane rotation where 3 of 5 feature marginals change.
    """
    feature_data = {f"f{j+1}": {} for j in range(5)}
    mu_y19 = [0.0, 0.0,  0.0,  0.0, 0.0]
    mu_y20 = [3.0, 3.0, -2.0,  0.5, 0.5]   # strong shift in f1,f2,f3

    for i in range(1, 25):
        mu = mu_y19 if i <= 12 else mu_y20
        tag = f"M{i:02d}"
        for j in range(5):
            feature_data[f"f{j+1}"][tag] = np.round(rng.normal(mu[j], 1.0, N_OBS), 4)

    return {name: _feature_df(vals) for name, vals in feature_data.items()}


# ************************************************************
# Master generator
# ************************************************************
def generate_multivariate(save: bool = True) -> dict:
    """
    Generate C23-C30. Returns dict {cid: {feature_name: DataFrame}}.
    Saves to DATASET_DIR/<CID>/<feature>.csv if save=True.
    """
    rng = np.random.default_rng(RANDOM_SEED + 200)
    os.makedirs(DATASET_DIR, exist_ok=True)

    generators = {
        "C23": generate_c23,
        "C24": generate_c24,
        "C25": generate_c25,
        "C26": generate_c26,
        "C27": generate_c27,
        "C28": generate_c28,
        "C29": generate_c29,
        "C30": generate_c30,
    }

    datasets = {}
    print("Generating multivariate datasets (C23-C30)...")
    for cid, fn in generators.items():
        features = fn(rng)
        datasets[cid] = features
        n_feat = len(features)
        if save:
            _save_multivariate(cid, features, DATASET_DIR)
        # Summary stats
        feat_names = list(features.keys())
        print(f"  ✔ {cid}  Features: {feat_names}  (n={n_feat})")
    print(f"\n✔ C23-C30 ready → {DATASET_DIR}\n")
    return datasets


def load_multivariate(cid: str, dataset_dir: str = None) -> dict:
    """
    Load a saved multivariate dataset from disk.
    Returns {feature_name: DataFrame}.
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    folder = os.path.join(dataset_dir, cid)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Multivariate dataset folder not found: {folder}")
    features = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".csv"):
            key = fname.replace(".csv", "")
            features[key] = pd.read_csv(os.path.join(folder, fname))
    return features


if __name__ == "__main__":
    generate_multivariate(save=True)
