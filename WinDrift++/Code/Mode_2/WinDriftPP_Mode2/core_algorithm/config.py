"""
config.py — WinDrift++ Configuration
=====================================
Central configuration for the WinDrift++ - Mode 2.
Edit this file to change algorithm parameters, paths, or dataset definitions.

Covers all 30 datasets:
  C1-C17   : Synthetic univariate
  C18-C22  : Real-world univariate (NOAA-inspired, Electricity, Covertype,
              Insects, Air Quality)
  C23-C25  : Multivariate real-world (3 datasets, 3-4 features each)
  C26-C30  : Multivariate synthetic (SEAa, SEAg, AGRa, AGRg, Hyperplane)
"""


# **** Shaping up provided code with parameters and correct indentation ****
WIN_LEV_SIZE  = [1, 3, 6, 12]
MAX_CYC_LEN   = 12
ALPHA         = 0.05
N_MONTHS      = 24
N_OBS         = 30
RANDOM_SEED   = 112

BASE_MU       = 5.00
BASE_SIGMA    = 1.00
MU_STEP       = 1.25
SIGMA_STEP    = 0.25

VARIANT_SCALE = {
    "C1":  (1.00, 1.00), "C2":  (0.90, 0.90), "C3":  (0.80, 0.80),
    "C4":  (1.10, 1.10), "C5":  (1.20, 1.20),
    "C6":  (1.00, 1.00), "C7":  (0.90, 0.90), "C8":  (0.80, 0.80),
    "C9":  (1.10, 1.10), "C10": (1.20, 1.20)}

C11_17_REF_MU    = 0.0
C11_17_REF_SIGMA = 1.0

# Dataset registry: one entry per dataset used in the experiment suite.
# Fields:
#   label        — short working label
#   group        — dataset category for grouping results
#   multivariate — True if dataset has multiple features (stored as per-feature CSVs)
#   n_features   — number of features (1 for univariate)
DATASET_REGISTRY = {
    "C1":  {"label": "Base (similar)",           "group": "synthetic-similar",      "multivariate": False, "n_features": 1},
    "C2":  {"label": "Scale -10% (similar)",     "group": "synthetic-similar",      "multivariate": False, "n_features": 1},
    "C3":  {"label": "Scale -20% (similar)",     "group": "synthetic-similar",      "multivariate": False, "n_features": 1},
    "C4":  {"label": "Scale +10% (similar)",     "group": "synthetic-similar",      "multivariate": False, "n_features": 1},
    "C5":  {"label": "Scale +20% (similar)",     "group": "synthetic-similar",      "multivariate": False, "n_features": 1},
    "C6":  {"label": "Base (dissimilar)",         "group": "synthetic-dissimilar",   "multivariate": False, "n_features": 1},
    "C7":  {"label": "Mean -10% (dissimilar)",   "group": "synthetic-dissimilar",   "multivariate": False, "n_features": 1},
    "C8":  {"label": "Mean -20% (dissimilar)",   "group": "synthetic-dissimilar",   "multivariate": False, "n_features": 1},
    "C9":  {"label": "Mean +10% (dissimilar)",   "group": "synthetic-dissimilar",   "multivariate": False, "n_features": 1},
    "C10": {"label": "Mean +20% (dissimilar)",   "group": "synthetic-dissimilar",   "multivariate": False, "n_features": 1},
    "C11": {"label": "Skewness shift",           "group": "synthetic-characteristic","multivariate": False, "n_features": 1},
    "C12": {"label": "Shape change",             "group": "synthetic-characteristic","multivariate": False, "n_features": 1},
    "C13": {"label": "Location shift",           "group": "synthetic-characteristic","multivariate": False, "n_features": 1},
    "C14": {"label": "Variance decrease",        "group": "synthetic-characteristic","multivariate": False, "n_features": 1},
    "C15": {"label": "Variance increase",        "group": "synthetic-characteristic","multivariate": False, "n_features": 1},
    "C16": {"label": "Outlier distribution",     "group": "synthetic-characteristic","multivariate": False, "n_features": 1},
    "C17": {"label": "Discretised (ties)",       "group": "synthetic-characteristic","multivariate": False, "n_features": 1},
    "C18": {"label": "Temperature (NOAA-style)", "group": "realworld-univariate",    "multivariate": False, "n_features": 1},
    "C19": {"label": "Electricity demand",       "group": "realworld-univariate",    "multivariate": False, "n_features": 1},
    "C20": {"label": "Forest elevation",         "group": "realworld-univariate",    "multivariate": False, "n_features": 1},
    "C21": {"label": "Insect population",        "group": "realworld-univariate",    "multivariate": False, "n_features": 1},
    "C22": {"label": "Air quality (PM2.5)",      "group": "realworld-univariate",    "multivariate": False, "n_features": 1},
    "C23": {"label": "Weather station",          "group": "realworld-multivariate",  "multivariate": True,  "n_features": 3},
    "C24": {"label": "Energy system",            "group": "realworld-multivariate",  "multivariate": True,  "n_features": 3},
    "C25": {"label": "Ecological sensors",       "group": "realworld-multivariate",  "multivariate": True,  "n_features": 4},
    "C26": {"label": "SEA abrupt (3-feat)",      "group": "synthetic-multivariate",  "multivariate": True,  "n_features": 3},
    "C27": {"label": "SEA gradual (3-feat)",     "group": "synthetic-multivariate",  "multivariate": True,  "n_features": 3},
    "C28": {"label": "Agrawal abrupt (3-feat)",  "group": "synthetic-multivariate",  "multivariate": True,  "n_features": 3},
    "C29": {"label": "Agrawal gradual (3-feat)", "group": "synthetic-multivariate",  "multivariate": True,  "n_features": 3},
    "C30": {"label": "Rotating hyperplane",      "group": "synthetic-multivariate",  "multivariate": True,  "n_features": 5}}

GROUPS = {
    "Exp1 Similar (C1-C5)":              [f"C{i}" for i in range(1, 6)],
    "Exp2 Dissimilar (C6-C10)":          [f"C{i}" for i in range(6, 11)],
    "Exp3 Characteristics (C11-C17)":    [f"C{i}" for i in range(11, 18)],
    "Exp4 Real Univariate (C18-C22)":    [f"C{i}" for i in range(18, 23)],
    "Exp5 Real Multivariate (C23-C25)":  [f"C{i}" for i in range(23, 26)],
    "Exp6 Synth Multivariate (C26-C30)": [f"C{i}" for i in range(26, 31)]}

UNIVARIATE_IDS   = [cid for cid, m in DATASET_REGISTRY.items() if not m["multivariate"]]
MULTIVARIATE_IDS = [cid for cid, m in DATASET_REGISTRY.items() if     m["multivariate"]]

import os
# BASE_DIR = parent of core_algorithm/ = project root
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(BASE_DIR, "Datasets")
RESULTS_DIR  = os.path.join(BASE_DIR, "Results")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR     = os.path.join(RESULTS_DIR, "logs")
