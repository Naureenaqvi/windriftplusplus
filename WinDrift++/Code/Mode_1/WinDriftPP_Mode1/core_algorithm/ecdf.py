"""
ecdf.py — Empirical Cumulative Distribution Function
=====================================================
Computes the Empirical Cumulative Distribution Function (ECDF).

    F̂n(x) = (number of records in window ≤ x) / n

Key behaviour:
  - Handles repeated values (ties) by assigning the HIGHEST proportion
    to all instances of the repeated observation (max-ECDF convention),
    assigns the maximum CDF value to all tied observations.
  - Evaluation is performed over the merged sorted union of both samples,
    which is the standard two-sample KS/ECDF framework.
"""


# **** Shaping up provided code with parameters and correct indentation ****
import numpy as np
from scipy import stats as st


def compute_ecdf_at_points(sample: np.ndarray, eval_points: np.ndarray,
                           round_decimals: int = None) -> np.ndarray:
    """
    Compute ECDF of `sample` evaluated at each value in `eval_points`.

    Uses scipy.stats.percentileofscore (kind='weak') which assigns the
    maximum CDF value to all tied observations.

    Parameters
    ----------
    sample       : 1-D array of observations forming the window
    eval_points  : ordered evaluation axis (typically the merged union of two samples)
    round_decimals: if set, rounds output to this many decimal places

    Returns
    -------
    ecdf_vals : array of shape (len(eval_points),)
    """
    ecdf_vals = np.array([
        st.percentileofscore(sample, x, kind="weak") / 100.0
        for x in eval_points
    ])
    if round_decimals is not None:
        ecdf_vals = np.round(ecdf_vals, round_decimals)
    return ecdf_vals


def compute_ecdf_pair(x_vals: np.ndarray, y_vals: np.ndarray,
                      round_decimals: int = None):
    """
    Compute ECDFs for both samples over their merged sorted union.

    Returns
    -------
    merged  : sorted union of x_vals and y_vals
    ecdf_x  : ECDF of x_vals evaluated at merged
    ecdf_y  : ECDF of y_vals evaluated at merged
    """
    merged = np.sort(np.concatenate([x_vals, y_vals]))
    ecdf_x = compute_ecdf_at_points(x_vals, merged, round_decimals)
    ecdf_y = compute_ecdf_at_points(y_vals, merged, round_decimals)
    return merged, ecdf_x, ecdf_y


def ecdf_differences(ecdf_x: np.ndarray, ecdf_y: np.ndarray):
    """
    Compute signed and absolute differences between two ECDFs.

    Returns
    -------
    diff     : F̂H(x) - F̂N(x)   (signed)
    abs_diff : |F̂H(x) - F̂N(x)| (absolute)
    """
    diff     = ecdf_x - ecdf_y
    abs_diff = np.abs(diff)
    return diff, abs_diff


if __name__ == "__main__":
    # Canonical example: ties handled by assigning max rank
    print("=== ECDF Example ===")
    x_unique = np.array([12.50, 17.80, 16.00, 16.70, 16.20, 16.80,
                         12.70, 16.90, 15.10, 13.00])
    x_sorted = np.sort(x_unique)
    print("Sorted x:", x_sorted)
    ecdf_unique = compute_ecdf_at_points(x_unique, x_sorted, round_decimals=1)
    print("ECDF (unique):", ecdf_unique)
    # Expected: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    x_ties = np.array([12.50, 12.70, 13.00, 15.10, 16.00,
                       16.00, 16.00, 16.80, 16.90, 17.80])
    x_ties_sorted = np.sort(x_ties)
    ecdf_ties = compute_ecdf_at_points(x_ties, x_ties_sorted, round_decimals=1)
    print("\nSorted x (with ties):", x_ties_sorted)
    print("ECDF (with ties):", ecdf_ties)
    # Expected: [0.1, 0.2, 0.3, 0.4, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0]
