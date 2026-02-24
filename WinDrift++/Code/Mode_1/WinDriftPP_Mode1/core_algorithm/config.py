"""
config.py — WinDrift++ configuration

Central configuration for the WinDrift++ - Mode1.
Edit this file to change algorithm parameters, paths, or dataset definitions.
"""


# **** Shaping up provided code with parameters and correct indentation ****
import os

# **** Paths ****
# BASE_DIR = parent of core_algorithm/ = project root
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(BASE_DIR, "Datasets")
RESULTS_DIR  = os.path.join(BASE_DIR, "Results")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR     = os.path.join(RESULTS_DIR, "logs")

# **** Algorithm parameters ****
WIN_LEV_SIZE = [1, 3, 6, 12]   # window sizes in months
MAX_CYC_LEN  = 12               # months per cycle (Y2019 = M01-M12, Y2020 = M13-M24)
ALPHA        = 0.05             # significance level for all statistical tests
N_OBS        = 30               # observations per monthly column
RANDOM_SEED  = 112

# **** Synthetic dataset generation parameters ****
BASE_MU      = 5.0              # Y2019 base mean
BASE_SIGMA   = 1.0              # Y2019 base standard deviation
MU_STEP      = 1.25             # monthly mean increment across Y2019
SIGMA_STEP   = 0.25             # monthly std increment across Y2019
VARIANT_SCALE = {               # Y2020 scaling factors per dataset variant
    "C1": (1.00, 1.00), "C2": (0.90, 0.90), "C3": (0.80, 0.80),
    "C4": (1.10, 1.10), "C5": (1.20, 1.20),
    "C6": (1.00, 1.00), "C7": (0.90, 1.00), "C8": (0.80, 1.00),
    "C9": (1.10, 1.00), "C10": (1.20, 1.00)}

# **** Dataset registry ****
# Fields:
#   label        — short working label
#   group        — dataset category for grouping results
#   multivariate — always False in Mode 1 (univariate only)
#   n_features   — always 1 in Mode 1
DATASET_REGISTRY = {
    "C1":  {"label": "Base (similar)",           "group": "synthetic-similar",       "multivariate": False, "n_features": 1},
    "C2":  {"label": "Scale -10% (similar)",     "group": "synthetic-similar",       "multivariate": False, "n_features": 1},
    "C3":  {"label": "Scale -20% (similar)",     "group": "synthetic-similar",       "multivariate": False, "n_features": 1},
    "C4":  {"label": "Scale +10% (similar)",     "group": "synthetic-similar",       "multivariate": False, "n_features": 1},
    "C5":  {"label": "Scale +20% (similar)",     "group": "synthetic-similar",       "multivariate": False, "n_features": 1},
    "C6":  {"label": "Base (dissimilar)",         "group": "synthetic-dissimilar",    "multivariate": False, "n_features": 1},
    "C7":  {"label": "Mean -10% (dissimilar)",   "group": "synthetic-dissimilar",    "multivariate": False, "n_features": 1},
    "C8":  {"label": "Mean -20% (dissimilar)",   "group": "synthetic-dissimilar",    "multivariate": False, "n_features": 1},
    "C9":  {"label": "Mean +10% (dissimilar)",   "group": "synthetic-dissimilar",    "multivariate": False, "n_features": 1},
    "C10": {"label": "Mean +20% (dissimilar)",   "group": "synthetic-dissimilar",    "multivariate": False, "n_features": 1},
    "C11": {"label": "Skewness shift",           "group": "synthetic-characteristic", "multivariate": False, "n_features": 1},
    "C12": {"label": "Shape change",             "group": "synthetic-characteristic", "multivariate": False, "n_features": 1},
    "C13": {"label": "Location shift",           "group": "synthetic-characteristic", "multivariate": False, "n_features": 1},
    "C14": {"label": "Variance decrease",        "group": "synthetic-characteristic", "multivariate": False, "n_features": 1},
    "C15": {"label": "Variance increase",        "group": "synthetic-characteristic", "multivariate": False, "n_features": 1},
    "C16": {"label": "Outlier distribution",     "group": "synthetic-characteristic", "multivariate": False, "n_features": 1},
    "C17": {"label": "Discretised (ties)",       "group": "synthetic-characteristic", "multivariate": False, "n_features": 1}}

GROUPS = {
    "Similar distributions (C1-C5)":          [f"C{i}" for i in range(1, 6)],
    "Dissimilar distributions (C6-C10)":       [f"C{i}" for i in range(6, 11)],
    "Statistical characteristics (C11-C17)":   [f"C{i}" for i in range(11, 18)]}

UNIVARIATE_IDS   = list(DATASET_REGISTRY.keys())
MULTIVARIATE_IDS = []
