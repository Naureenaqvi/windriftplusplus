"""
evaluation.py — Detection Statistics
=====================================
Summarises what WinDrift++ detected across each dataset's comparison windows.

Reports raw detection counts and rates — no ground truth required.
"""


# **** Shaping up provided code with parameters and correct indentation ****
import numpy as np
import pandas as pd
import math


def summarise_dataset(dataset_id: str,
                      results_df: pd.DataFrame,
                      mode: int = 2) -> dict:
    """
    Summarise detection results for one dataset.

    Returns counts and rates from Mode II (Corresponding) comparisons:
      total_comparisons  — number of window pairs evaluated
      flags_raised       — number flagged as drift by majority vote
      flag_rate          — flags_raised / total_comparisons
      first_flag_dbcount — dbcount of first drift flag (None if none raised)
      MTD                — mean dbcount of all flagged comparisons
      per-test counts    — how many comparisons each individual test flagged
    """
    sub = results_df[results_df["mode"] == mode]
    total = len(sub)
    flags = sub["majority_flag"].values
    flagged = int(np.sum(flags == 1))
    flag_rate = round(flagged / total, 4) if total > 0 else 0.0

    flagged_rows = sub[sub["majority_flag"] == 1]["dbcount"].values
    first_flag = int(flagged_rows.min()) if len(flagged_rows) > 0 else None
    mtd = round(float(np.mean(flagged_rows)), 2) if len(flagged_rows) > 0 else None

    row = {
        "dataset_id":        dataset_id,
        "total_comparisons": total,
        "flags_raised":      flagged,
        "flag_rate":         flag_rate,
        "first_flag_dbcount": first_flag,
        "MTD":               mtd,
    }

    # Per-test flag counts
    for test in ["ks", "kuiper", "cvm", "ad", "emd"]:
        col = f"{test}_flag"
        if col in sub.columns:
            row[f"{test}_flags"] = int(sub[col].sum())

    return row


def build_summary_table(all_summaries: list) -> pd.DataFrame:
    """Assemble a summary DataFrame from per-dataset result dicts."""
    return pd.DataFrame(all_summaries)


def per_test_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    How many drift flags each individual test raised across all Mode II comparisons.
    Useful for ablation analysis.
    """
    sub = results_df[results_df["mode"] == 2]
    rows = []
    for test in ["ks", "kuiper", "cvm", "ad", "emd"]:
        col  = f"{test}_flag"
        n_d  = int(sub[col].sum())
        n_nd = int((sub[col] == 0).sum())
        rows.append({"test": test.upper(), "flagged": n_d, "not_flagged": n_nd})
    rows.append({
        "test":        "WD++ (Majority)",
        "flagged":     int(sub["majority_flag"].sum()),
        "not_flagged": int((sub["majority_flag"] == 0).sum()),
    })
    return pd.DataFrame(rows)
