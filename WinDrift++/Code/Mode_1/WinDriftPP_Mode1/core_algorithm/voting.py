"""
voting.py — Majority Voting Decision Mechanism
===============================================
Implements the Decide component of WinDrift++: aggregates per-test drift flags
into a single drift decision using configurable voting strategies.

The candidate set B = {D, D̄} where D=drift, D̄=no-drift.

With B = 5 tests (odd number — recommended by paper):
  - If votes_for_drift >  2 → D  (Drift)
  - If votes_for_drift <  3 → D̄ (No Drift)
  - (Tie cannot occur with odd number of tests)

With even B:
  - Tie (e.g. 2-2) → 'U' (Unsure)

Additional voting strategies for sensitivity analysis:
  majority, plurality, borda_count, plurality_elimination, pairwise
"""


# **** Shaping up provided code with parameters and correct indentation ****
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from statistical_tests import TestResult


@dataclass
class VoteResult:
    """Container for the final drift decision after voting."""
    drift_flag  : int          # 1=Drift, 0=No Drift, -1=Unsure
    label       : str          # 'D', 'D̄', or 'U'
    votes_drift : int
    votes_nodrift : int
    total_tests : int
    method      : str
    individual  : List[int]    # drift flags from each test

    def __repr__(self):
        return (f"[{self.method}] Drift={self.votes_drift}/{self.total_tests} "
                f"NoD={self.votes_nodrift}/{self.total_tests}  → {self.label}")


def _extract_flags(test_results: List[TestResult]) -> List[int]:
    return [r.drift_flag for r in test_results]


# ************************************************************
# Default: Majority Voting — requires ≥3/5 tests to agree
# ************************************************************

def majority_vote(test_results: List[TestResult]) -> VoteResult:
    """
    Drift if strictly more than half of B tests detect drift.
    With B=5 (odd): needs ≥ 3 votes for drift.
    With B=even: tie → 'U' (Unsure).
    """
    flags = _extract_flags(test_results)
    B     = len(flags)
    v_d   = sum(flags)
    v_nd  = B - v_d

    if v_d > v_nd:
        return VoteResult(1, "D",  v_d, v_nd, B, "Majority", flags)
    elif v_nd > v_d:
        return VoteResult(0, "D̄", v_d, v_nd, B, "Majority", flags)
    else:
        return VoteResult(-1, "U", v_d, v_nd, B, "Majority", flags)


# ************************************************************
# Plurality Voting
# ************************************************************

def plurality_vote(test_results: List[TestResult]) -> VoteResult:
    """
    Outcome with the most votes wins (same as majority for binary candidates,
    but ties are broken differently — here we default to D̄ on tie).
    For binary {D, D̄} this is identical to majority unless tied.
    """
    flags = _extract_flags(test_results)
    B     = len(flags)
    v_d   = sum(flags)
    v_nd  = B - v_d

    if v_d >= v_nd:      # Note: ties default to Drift in plurality
        return VoteResult(1 if v_d > 0 else 0, "D" if v_d > 0 else "D̄",
                          v_d, v_nd, B, "Plurality", flags)
    return VoteResult(0, "D̄", v_d, v_nd, B, "Plurality", flags)


# ************************************************************
# Borda Count
# ************************************************************

def borda_count_vote(test_results: List[TestResult]) -> VoteResult:
    """
    Borda Count: each test's p-value rank contributes a score.
    Tests with lower p-values (stronger evidence for drift) rank higher.
    Score for drift = sum of inverse p-values (proxy for Borda ranking).
    """
    flags  = _extract_flags(test_results)
    B      = len(flags)
    v_d    = sum(flags)
    v_nd   = B - v_d

    # Use p-values for ranking: lower p → more confident in drift
    p_vals = np.array([r.p_value if not np.isnan(r.p_value) else 1.0
                       for r in test_results])

    # Borda rank: rank from lowest p-value (most significant) to highest
    ranks = np.argsort(np.argsort(p_vals))  # 0=lowest p, B-1=highest p
    borda_scores_drift   = sum((B - 1 - ranks[i]) for i, f in enumerate(flags) if f == 1)
    borda_scores_nodrift = sum((B - 1 - ranks[i]) for i, f in enumerate(flags) if f == 0)

    if borda_scores_drift > borda_scores_nodrift:
        return VoteResult(1, "D",  v_d, v_nd, B, "Borda", flags)
    elif borda_scores_nodrift > borda_scores_drift:
        return VoteResult(0, "D̄", v_d, v_nd, B, "Borda", flags)
    else:
        return VoteResult(-1, "U", v_d, v_nd, B, "Borda", flags)


# ************************************************************
# Plurality with Elimination
# ************************************************************

def plurality_elimination_vote(test_results: List[TestResult]) -> VoteResult:
    """
    Iterative elimination: remove the least-confident test until majority.
    Confidence measured by p-value distance from alpha (0.05).
    """
    flags    = _extract_flags(test_results)
    B        = len(flags)
    v_d      = sum(flags)
    v_nd     = B - v_d
    remaining = list(range(B))
    current_flags = flags[:]

    while len(remaining) > 1:
        curr_vd  = sum(current_flags[i] for i in remaining)
        curr_vnd = len(remaining) - curr_vd
        if curr_vd != curr_vnd:
            break
        # Eliminate the test with p-value closest to 0.5 (least informative)
        p_vals = [abs(test_results[i].p_value - 0.5) for i in remaining]
        least_info = remaining[np.argmin(p_vals)]
        remaining.remove(least_info)

    final_vd  = sum(current_flags[i] for i in remaining)
    final_vnd = len(remaining) - final_vd

    if final_vd > final_vnd:
        return VoteResult(1, "D",  v_d, v_nd, B, "Elimination", flags)
    elif final_vnd > final_vd:
        return VoteResult(0, "D̄", v_d, v_nd, B, "Elimination", flags)
    else:
        return VoteResult(-1, "U", v_d, v_nd, B, "Elimination", flags)


# ************************************************************
# Pairwise Comparison
# ************************************************************

def pairwise_vote(test_results: List[TestResult]) -> VoteResult:
    """
    Condorcet-style pairwise comparison: each pair of tests compared,
    the outcome with most pairwise wins is selected.
    For binary {D, D̄}: equivalent to majority but more robust to ties.
    """
    flags = _extract_flags(test_results)
    B     = len(flags)
    v_d   = sum(flags)
    v_nd  = B - v_d

    # With binary outcomes, pairwise reduces to: each (D,D̄) pair
    # D wins pairwise if D-voter p-value < D̄-voter p-value
    drift_score   = 0
    nodrift_score = 0
    drift_idx   = [i for i, f in enumerate(flags) if f == 1]
    nodrift_idx = [i for i, f in enumerate(flags) if f == 0]

    for di in drift_idx:
        for ni in nodrift_idx:
            pd = test_results[di].p_value
            pn = test_results[ni].p_value
            if pd <= pn:
                drift_score += 1
            else:
                nodrift_score += 1

    if v_d == 0:
        return VoteResult(0, "D̄", v_d, v_nd, B, "Pairwise", flags)
    if v_nd == 0:
        return VoteResult(1, "D",  v_d, v_nd, B, "Pairwise", flags)

    if drift_score > nodrift_score:
        return VoteResult(1, "D",  v_d, v_nd, B, "Pairwise", flags)
    elif nodrift_score > drift_score:
        return VoteResult(0, "D̄", v_d, v_nd, B, "Pairwise", flags)
    else:
        return VoteResult(-1, "U", v_d, v_nd, B, "Pairwise", flags)


# ************************************************************
# Run all voting methods and return dict
# ************************************************************

def all_voting_methods(test_results: List[TestResult]) -> dict:
    """Returns results from all 5 voting methods for comparison."""
    return {
        "majority"    : majority_vote(test_results),
        "plurality"   : plurality_vote(test_results),
        "borda"       : borda_count_vote(test_results),
        "elimination" : plurality_elimination_vote(test_results),
        "pairwise"    : pairwise_vote(test_results),
    }
