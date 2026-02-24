"""
windrift_pp_multivariate.py — Multivariate WinDrift++ Extension
================================================================
Extends WinDrift++ to handle multivariate datasets (C23-C30).

Algorithm:
  1. Run WinDrift++ independently on each feature (using the same
     windowing parameters as univariate detection).
  2. At each (dbcount, win_level) combination, collect the majority_flag
     from all features.
  3. Apply a FEATURE-LEVEL majority vote: if >= ceil(n_features/2) features
     flag drift, the dataset-level decision is DRIFT.

Multivariate extension of WinDrift++:
of the WinDrift++ paper, which notes that the algorithm can be extended
to multivariate data by applying per-variable detection followed by an
aggregation step.

Returns a standard results DataFrame with the same schema as univariate
detection, plus additional columns:
  - feature_flags      : dict of {feature_name: flag}
  - features_drifting  : count of features flagging drift
  - n_features         : total number of features
"""


# **** Shaping up provided code with parameters and correct indentation ****
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from windrift_pp import WinDriftPP


class WinDriftPPMultivariate:
    """
    Multivariate WinDrift++ detector.

    Parameters
    ----------
    win_lev_size : list of int
    max_cyc_len  : int
    alpha        : float
    verbose      : bool
    """

    def __init__(self, win_lev_size=None, max_cyc_len=12, alpha=0.05, verbose=False):
        from config import WIN_LEV_SIZE
        self.win_lev_size = win_lev_size or WIN_LEV_SIZE
        self.max_cyc_len  = max_cyc_len
        self.alpha        = alpha
        self.verbose      = verbose
        self._results_df  = None

    def detect(self, features: dict, dataset_id: str) -> pd.DataFrame:
        """
        Run per-feature WinDrift++ and aggregate.

        Parameters
        ----------
        features   : dict {feature_name: DataFrame}
                     Each DataFrame has shape (N_OBS, N_MONTHS), same as univariate.
        dataset_id : str  e.g. "C23"

        Returns
        -------
        DataFrame with one row per (mode, dbcount, win_level) combination,
        including per-feature flags and the aggregated multivariate majority flag.
        """
        n_features = len(features)
        if n_features == 0:
            raise ValueError("features dict is empty")

        # Run WD++ on each feature independently
        per_feature_results = {}
        for feat_name, df in features.items():
            detector = WinDriftPP(
                win_lev_size=self.win_lev_size,
                max_cyc_len=self.max_cyc_len,
                alpha=self.alpha,
                verbose=False,
            )
            detector.detect(df, dataset_id=f"{dataset_id}_{feat_name}")
            per_feature_results[feat_name] = detector.to_dataframe()

        # Merge on the index keys (mode, dbcount, win_level)
        key_cols = ["mode", "mode_name", "dbcount", "win_level", "label_h", "label_n"]

        # Use first feature as skeleton
        first_name = list(per_feature_results.keys())[0]
        base_df = per_feature_results[first_name][key_cols].copy()
        base_df["dataset_id"] = dataset_id

        # Collect majority_flag per feature
        flag_cols = {}
        for feat_name, res_df in per_feature_results.items():
            merged = base_df.merge(
                res_df[key_cols + ["majority_flag"]].rename(
                    columns={"majority_flag": f"flag_{feat_name}"}
                ),
                on=key_cols, how="left"
            )
            flag_cols[f"flag_{feat_name}"] = merged[f"flag_{feat_name}"]

        for col, series in flag_cols.items():
            base_df[col] = series.values

        # Feature-level vote: count drifting features (flag==1)
        flag_col_names = [f"flag_{fn}" for fn in features.keys()]
        drift_counts = base_df[flag_col_names].apply(
            lambda row: (row == 1).sum(), axis=1
        )
        base_df["features_drifting"] = drift_counts
        base_df["n_features"]        = n_features

        # Aggregate majority vote: drift if > half the features flag drift
        threshold = n_features / 2.0
        base_df["majority_flag"] = drift_counts.apply(
            lambda c: 1 if c > threshold else (-1 if c == threshold and n_features % 2 == 0 else 0)
        )

        if self.verbose:
            n_drift = (base_df["majority_flag"] == 1).sum()
            print(f"  [{dataset_id}] {n_drift}/{len(base_df)} positions flagged drift "
                  f"(across {n_features} features)")

        self._results_df = base_df
        return base_df

    def to_dataframe(self) -> pd.DataFrame:
        return self._results_df
