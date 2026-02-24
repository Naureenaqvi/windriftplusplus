"""
windrift_pp.py — WinDrift++ Core Engine
========================================
Core WinDrift++ detection engine.

Three steps:
  Step 1: Configure   — set winLevSize, maxCycLen, dataBlocks
  Step 2: Assess Drift — sliding windows → ECDF → 5 statistical tests
  Step 3: Make Decision — majority voting over test results

Two exclusive modes:
  Mode I  (Consecutive)  : compare adjacent windows within a cycle
  Mode II (Corresponding): compare same-period windows across cycles

Window indexing (1-based column positions):
  Mode I  : dbcount ≤ maxCycLen
  Mode II : dbcount > maxCycLen  (but ≥ maxCycLen to also include boundary)

k1 = i*j          → WH1 start for Mode I
l1 = i*(j+1)      → WN1 start for Mode I
k2 = (m-1)*o + k1 → WH2 start for Mode II  (previous cycle)
l2 = m*o    + k1  → WN2 start for Mode II  (current cycle)
"""


# **** Shaping up provided code with parameters and correct indentation ****
import numpy as np
import pandas as pd
import time
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))
from config import WIN_LEV_SIZE, MAX_CYC_LEN, ALPHA, LOGS_DIR
from statistical_tests import run_all_tests, TestResult
from voting import majority_vote, all_voting_methods, VoteResult


@dataclass
class DriftRecord:
    """Single drift detection event record."""
    dataset_id   : str
    mode         : int            # 1=Consecutive, 2=Corresponding
    mode_name    : str
    dbcount      : int
    win_level    : int
    label_h      : str            # historical window label
    label_n      : str            # new window label
    n_h          : int            # size of historical window
    n_n          : int            # size of new window
    # Per-test results
    ks_stat      : float
    ks_crit      : float
    ks_p         : float
    ks_flag      : int
    kuiper_stat  : float
    kuiper_crit  : float
    kuiper_p     : float
    kuiper_flag  : int
    cvm_stat     : float
    cvm_crit     : float
    cvm_p        : float
    cvm_flag     : int
    ad_stat      : float
    ad_crit      : float
    ad_p         : float
    ad_flag      : int
    emd_stat     : float
    emd_crit     : float
    emd_p        : float
    emd_flag     : int
    # Voting result
    votes_drift  : int
    votes_nodrift: int
    majority_flag: int            # final WD++ decision (majority)
    majority_label: str


# ************************************************************
# Window builder utilities (mirrors original windrift.py logic)
# ************************************************************

def _build_window(df: pd.DataFrame, columns: list,
                  col_indices: list) -> np.ndarray:
    """Concatenate observations from the specified column indices."""
    data = np.array([])
    for idx in col_indices:
        if 0 <= idx < len(columns):
            data = np.concatenate([data, df.iloc[:, idx].values.astype(float)])
    return data


def _window_label(columns: list, col_indices: list,
                  sep: str = ",") -> str:
    labels = [columns[i] for i in col_indices if 0 <= i < len(columns)]
    return sep.join(labels)


def _consecutive_indices(dbcount: int, win_level: int,
                         maxcol: int, max_cyc_len: int):
    """
    Returns (hist_indices, new_indices) for Mode I.
    Mirrors the windriftconsecutive() logic from windrift_v1.py.
    """
    if win_level == 1:
        if dbcount < maxcol and dbcount < max_cyc_len:
            return [dbcount - 1], [dbcount]
    elif dbcount % win_level == 0 and dbcount > win_level:
        start_a = max(0, dbcount - 2 * win_level)
        end_a   = start_a + win_level
        start_b = end_a
        end_b   = start_b + win_level

        hist_idx = list(range(start_a, min(end_a, maxcol)))
        new_idx  = list(range(start_b, min(end_b, maxcol)))
        if hist_idx and new_idx:
            return hist_idx, new_idx
    return [], []


def _corresponding_indices(dbcount: int, win_level: int,
                           maxcol: int, max_cyc_len: int):
    """
    Returns (hist_indices, new_indices) for Mode II.
    Mirrors the windriftcorresponding() logic from windrift_v1.py.
    """
    local = dbcount - max_cyc_len

    if win_level == 1:
        if dbcount < maxcol:
            yr1_idx = dbcount - max_cyc_len
            return [yr1_idx], [dbcount]
    elif local % win_level == 0 and local >= win_level:
        start_a = max(0, dbcount - max_cyc_len - win_level + 1) - 1
        if start_a < 0:
            start_a = 0
        end_a   = start_a + win_level
        start_b = start_a + max_cyc_len
        end_b   = start_b + win_level

        hist_idx = list(range(start_a, min(end_a, maxcol)))
        new_idx  = list(range(start_b, min(end_b, maxcol)))
        if hist_idx and new_idx:
            return hist_idx, new_idx
    return [], []


# ************************************************************
# WinDrift++ main class
# ************************************************************

class WinDriftPP:
    """
    WinDrift++ concept drift detector.

    Parameters
    ----------
    win_lev_size : list of window sizes, default [1,3,6,12]
    max_cyc_len  : cycle boundary (= win_Max), default 12
    alpha        : significance level, default 0.05
    verbose      : print progress if True
    """

    def __init__(self,
                 win_lev_size: list = None,
                 max_cyc_len:  int  = MAX_CYC_LEN,
                 alpha:        float = ALPHA,
                 verbose:      bool  = False):
        self.win_lev_size  = sorted(win_lev_size or WIN_LEV_SIZE)
        self.max_cyc_len   = max_cyc_len
        self.alpha         = alpha
        self.verbose       = verbose
        self.records_: List[DriftRecord] = []

    def detect(self, df: pd.DataFrame, dataset_id: str = "??") -> List[DriftRecord]:
        """
        Run WinDrift++ over the full dataset.
        Returns list of DriftRecord for every comparison made.
        """
        self.records_ = []
        columns = list(df.columns)
        maxcol  = len(columns)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  WinDrift++ | {dataset_id} | cols={maxcol} | rows={len(df)}")
            print(f"  winLevSize={self.win_lev_size} | maxCycLen={self.max_cyc_len}")
            print(f"{'='*70}")
            print("  ── Mode I: Consecutive ──────────────────────────────")

        # **** Mode I: Consecutive (dbcount 1 … max_cyc_len) ****
        for dbcount in range(1, self.max_cyc_len + 1):
            for win_level in self.win_lev_size:
                h_idx, n_idx = _consecutive_indices(
                    dbcount, win_level, maxcol, self.max_cyc_len
                )
                if not h_idx or not n_idx:
                    continue
                x_h = _build_window(df, columns, h_idx)
                x_n = _build_window(df, columns, n_idx)
                if len(x_h) == 0 or len(x_n) == 0:
                    continue
                lbl_h = _window_label(columns, h_idx, sep=",")
                lbl_n = _window_label(columns, n_idx, sep=",")
                self._run_and_record(
                    dataset_id, 1, "Consecutive",
                    dbcount, win_level, x_h, x_n, lbl_h, lbl_n
                )

        if self.verbose:
            print("  ── Mode II: Corresponding ───────────────────────────")

        # **** Mode II: Corresponding (dbcount max_cyc_len … maxcol) ****
        for dbcount in range(self.max_cyc_len, maxcol + 1):
            for win_level in self.win_lev_size:
                h_idx, n_idx = _corresponding_indices(
                    dbcount, win_level, maxcol, self.max_cyc_len
                )
                if not h_idx or not n_idx:
                    continue
                x_h = _build_window(df, columns, h_idx)
                x_n = _build_window(df, columns, n_idx)
                if len(x_h) == 0 or len(x_n) == 0:
                    continue
                lbl_h = _window_label(columns, h_idx, sep=":")
                lbl_n = _window_label(columns, n_idx, sep=":")
                self._run_and_record(
                    dataset_id, 2, "Corresponding",
                    dbcount, win_level, x_h, x_n, lbl_h, lbl_n
                )

        return self.records_

    def _run_and_record(self, dataset_id, mode, mode_name,
                        dbcount, win_level, x_h, x_n, lbl_h, lbl_n):
        """Run all 5 tests + voting, store DriftRecord."""
        test_results = run_all_tests(x_h, x_n, self.alpha)
        vote = majority_vote(test_results)

        ks, ku, cv, ad, em = test_results

        rec = DriftRecord(
            dataset_id   = dataset_id,
            mode         = mode,
            mode_name    = mode_name,
            dbcount      = dbcount,
            win_level    = win_level,
            label_h      = lbl_h,
            label_n      = lbl_n,
            n_h          = len(x_h),
            n_n          = len(x_n),
            ks_stat      = round(ks.stat, 4),
            ks_crit      = round(ks.crit, 4),
            ks_p         = round(ks.p_value, 4),
            ks_flag      = ks.drift_flag,
            kuiper_stat  = round(ku.stat, 4),
            kuiper_crit  = round(ku.crit, 4),
            kuiper_p     = round(ku.p_value, 4),
            kuiper_flag  = ku.drift_flag,
            cvm_stat     = round(cv.stat, 4),
            cvm_crit     = round(cv.crit, 4),
            cvm_p        = round(cv.p_value, 4),
            cvm_flag     = cv.drift_flag,
            ad_stat      = round(ad.stat, 4),
            ad_crit      = round(ad.crit, 4),
            ad_p         = round(ad.p_value, 4),
            ad_flag      = ad.drift_flag,
            emd_stat     = round(em.stat, 4),
            emd_crit     = round(em.crit, 4),
            emd_p        = round(em.p_value, 4),
            emd_flag     = em.drift_flag,
            votes_drift  = vote.votes_drift,
            votes_nodrift= vote.votes_nodrift,
            majority_flag= vote.drift_flag,
            majority_label= vote.label,
        )
        self.records_.append(rec)

        if self.verbose:
            arrow = "⚠ DRIFT" if vote.drift_flag == 1 else "  no drift"
            print(f"  [{mode_name[:4]:4s}, W={win_level:2d}] "
                  f"{lbl_h} vs {lbl_n}  "
                  f"Votes: {vote.votes_drift}/5  → {arrow}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to a pandas DataFrame."""
        return pd.DataFrame([vars(r) for r in self.records_])
