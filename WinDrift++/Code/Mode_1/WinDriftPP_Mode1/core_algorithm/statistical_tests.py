"""
statistical_tests.py — Five ECDF-based Statistical Distance Tests
=================================================================
Implements all five statistical tests defined in WinDrift++ (Equations 3–7):

  (3) KS    — Kolmogorov-Smirnov     : max |F̂H - F̂N|
  (4) Kuiper — Kuiper test            : max(F̂H-F̂N) + |min(F̂H-F̂N)|
  (5) CVM   — Cramér-von Mises       : Σ |F̂H(x) - F̂N(x)|
  (6) AD    — Anderson-Darling       : Σ |F̂H-F̂N| / [Ĉ(1-Ĉ)]
  (7) EMD   — Earth Mover's Distance : ∫ |F̂H(x) - F̂N(x)| dx  (Wasserstein-1)

Each test returns:
  TestResult(stat, crit, p_value, drift_flag)

where drift_flag = 1 if Ds > Dc (or p_value < alpha), else 0.

Critical value conventions:
  KS    : D_crit = 1.36 * √(1/n + 1/m)        (two-tailed, α≈0.05)
  Kuiper: V_crit ≈ (1.747 + 0.11/√λ) / √λ     (Kuiper approximation)
  CVM   : p-value from scipy.stats.cramervonmises_2samp
  AD    : p-value from scipy.stats.anderson_ksamp
  EMD   : permutation test (500 permutations) for critical value
"""


# **** Shaping up provided code with parameters and correct indentation ****
import numpy as np
from scipy import stats as st
from scipy.stats import wasserstein_distance
from dataclasses import dataclass
from typing import Optional
import warnings

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import ALPHA
from ecdf import compute_ecdf_pair, ecdf_differences

warnings.filterwarnings("ignore")


@dataclass
class TestResult:
    """Container for a single statistical test outcome."""
    test_name  : str
    stat       : float   # distance statistic Ds
    crit       : float   # critical value Dc (nan if p-value based)
    p_value    : float   # p-value (nan if not available)
    drift_flag : int     # 1 = drift, 0 = no drift
    alpha      : float   = ALPHA

    def __repr__(self):
        flag_str = "DRIFT" if self.drift_flag else "no drift"
        return (f"[{self.test_name:7s}] Ds={self.stat:.4f}  "
                f"Dc={self.crit:.4f}  p={self.p_value:.4f}  → {flag_str}")


# ************************************************************
# 1. Kolmogorov-Smirnov (KS) Test
# ************************************************************

def ks_test(x_hist: np.ndarray, x_new: np.ndarray,
            alpha: float = ALPHA) -> TestResult:
    """
    KS = max_{x∈R} |F̂H(x) − F̂N(x)|
    D_crit = 1.36 × √(1/n + 1/m)   [two-sided, α≈0.05, original WD formula]
    Supplemented with scipy p-value for reference.
    """
    n, m = len(x_hist), len(x_new)
    _, ecdf_h, ecdf_n = compute_ecdf_pair(x_hist, x_new)
    _, abs_diff = ecdf_differences(ecdf_h, ecdf_n)
    ks_stat = float(np.max(abs_diff))

    # Critical value from original WinDrift paper
    d_crit = 1.36 * np.sqrt(1.0/n + 1.0/m)

    # Cross-validate with scipy
    _, p_val = st.ks_2samp(x_hist, x_new)

    drift_flag = 1 if ks_stat >= d_crit else 0
    return TestResult("KS", ks_stat, d_crit, p_val, drift_flag, alpha)


# ************************************************************
# 2. Kuiper Test
# ************************************************************

def kuiper_test(x_hist: np.ndarray, x_new: np.ndarray,
                alpha: float = ALPHA) -> TestResult:
    """
    Kuiper = max_{x}(F̂H(x) − F̂N(x)) + max_{x}(F̂N(x) − F̂H(x))
           = D⁺ + D⁻   (sum of max positive and max negative deviations)

    Critical value uses the Kuiper distribution approximation:
      V_crit ≈ (1.747 + 0.11/√λ + 0.1/λ) / √λ
      where λ = √(n×m/(n+m))
    Reference: Press et al., Numerical Recipes, §14.3
    """
    n, m = len(x_hist), len(x_new)
    _, ecdf_h, ecdf_n = compute_ecdf_pair(x_hist, x_new)
    diff, _ = ecdf_differences(ecdf_h, ecdf_n)

    d_plus  = float(np.max(diff))
    d_minus = float(np.max(-diff))
    kuiper_stat = d_plus + d_minus

    # Effective sample size λ for critical value
    lam = np.sqrt((n * m) / (n + m))
    # Kuiper critical value at α=0.05 (Press et al., 1992)
    v_crit = (1.747 + 0.11 / lam + 0.10 / (lam ** 2)) / lam

    # P-value via asymptotic Kuiper distribution
    # V_observed * (sqrt(N_eff) + 0.155 + 0.24/sqrt(N_eff))
    N_eff = lam
    lam_obs = (kuiper_stat) * (N_eff + 0.155 + 0.24 / N_eff)
    # Kuiper CDF approximation: P(V > x) ≈ 2Σ(4j²x²-1)exp(-2j²x²)
    p_val = 0.0
    for j in range(1, 100):
        term = (4 * j**2 * lam_obs**2 - 1) * np.exp(-2 * j**2 * lam_obs**2)
        p_val += term
        if abs(term) < 1e-8:
            break
    p_val = min(max(2 * p_val, 0.0), 1.0)

    drift_flag = 1 if kuiper_stat >= v_crit else 0
    return TestResult("Kuiper", kuiper_stat, v_crit, p_val, drift_flag, alpha)


# ************************************************************
# 3. Cramér-von Mises (CVM) Test
# ************************************************************

def cvm_test(x_hist: np.ndarray, x_new: np.ndarray,
             alpha: float = ALPHA) -> TestResult:
    """
    CVM = Σ_{x∈X} |F̂H(x) − F̂N(x)|    (sum of absolute ECDF differences)
    Note: The paper uses Σ (not Σ²), reflecting the CVM spirit
    but computed as sum of absolute differences over merged support.

    Critical value: uses scipy.stats.cramervonmises_2samp p-value.
    The CVM statistic from scipy uses the squared version:
      ω² = (n×m)/(n+m)² Σ(F̂H - F̂N)²
    We use Σ|F_n(x) - G_m(x)| as the primary statistic, scipy's p-value for decision.
    """
    _, ecdf_h, ecdf_n = compute_ecdf_pair(x_hist, x_new)
    _, abs_diff = ecdf_differences(ecdf_h, ecdf_n)

    # Paper's definition: Σ |F̂H - F̂N|
    cvm_stat = float(np.sum(abs_diff))

    # Use scipy for rigorous p-value (standard CVM squared form)
    result = st.cramervonmises_2samp(x_hist, x_new)
    p_val = result.pvalue

    # Critical value: approximate from scipy's statistic at α=0.05
    # Derive empirical threshold for the Σ|.| statistic
    # using the relationship to standard CVM
    n, m = len(x_hist), len(x_new)
    # Approximate: Dc for sum|.| ≈ 3.0 × √(n+m)/(n*m) at α=0.05
    d_crit = 3.0 * np.sqrt((n + m) / (n * m))

    drift_flag = 1 if (p_val < alpha) else 0
    return TestResult("CVM", cvm_stat, d_crit, p_val, drift_flag, alpha)


# ************************************************************
# 4. Anderson-Darling (AD) Test
# ************************************************************

def ad_test(x_hist: np.ndarray, x_new: np.ndarray,
            alpha: float = ALPHA) -> TestResult:
    """
    AD = Σ_{x∈X} |F̂H(x) − F̂N(x)| / [Ĉ(x)(1 − Ĉ(x))]

    where Ĉ(x) is the ECDF of the combined distribution W[H,N].

    Critical value and p-value from scipy.stats.anderson_ksamp.
    """
    # Combined ECDF Ĉ(x)
    combined = np.sort(np.concatenate([x_hist, x_new]))
    n_comb   = len(combined)

    _, ecdf_h, ecdf_n = compute_ecdf_pair(x_hist, x_new)

    # Combined ECDF evaluated at merged points
    ecdf_c = np.array([
        st.percentileofscore(combined, v, kind="weak") / 100.0
        for v in combined
    ])
    # Clip to avoid division by zero at boundaries
    ecdf_c_safe = np.clip(ecdf_c, 1e-6, 1 - 1e-6)

    _, abs_diff = ecdf_differences(ecdf_h, ecdf_n)
    weights  = ecdf_c_safe * (1.0 - ecdf_c_safe)
    ad_stat  = float(np.sum(abs_diff / weights))

    # Use scipy's Anderson-Darling 2-sample test for p-value
    try:
        result = st.anderson_ksamp([x_hist, x_new])
        p_val  = float(result.significance_level / 100.0)   # scipy returns %
        ad_sci = float(result.statistic)
        # Critical value at α=0.05 from scipy's table
        d_crit = float(result.critical_values[2])  # 5% level
    except Exception:
        p_val  = np.nan
        d_crit = np.nan

    drift_flag = 1 if (not np.isnan(p_val) and p_val < alpha) else 0
    return TestResult("AD", ad_stat, d_crit if not np.isnan(d_crit) else 0.0,
                      p_val if not np.isnan(p_val) else 1.0, drift_flag, alpha)


# ************************************************************
# 5. Earth Mover's Distance (EMD / Wasserstein-1)
# ************************************************************

def emd_test(x_hist: np.ndarray, x_new: np.ndarray,
             alpha: float = ALPHA,
             n_permutations: int = 500) -> TestResult:
    """
    EMD = ∫_{-∞}^{∞} |F̂H(x) − F̂N(x)| dx   (L1 Wasserstein / Earth Mover)

    scipy.stats.wasserstein_distance computes this exactly for 1D distributions.

    Critical value: permutation test — shuffle labels n_permutations times
    and take the (1-alpha) quantile of the null distribution.

    n_permutations=500 balances accuracy and speed.
    """
    emd_stat = float(wasserstein_distance(x_hist, x_new))

    # Permutation test for significance
    rng      = np.random.default_rng(42)
    combined = np.concatenate([x_hist, x_new])
    n        = len(x_hist)
    null_dist = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_dist[i] = wasserstein_distance(perm[:n], perm[n:])

    d_crit = float(np.quantile(null_dist, 1 - alpha))
    p_val  = float(np.mean(null_dist >= emd_stat))

    drift_flag = 1 if emd_stat >= d_crit else 0
    return TestResult("EMD", emd_stat, d_crit, p_val, drift_flag, alpha)


# ************************************************************
# Composite: run all 5 tests
# ************************************************************

def run_all_tests(x_hist: np.ndarray, x_new: np.ndarray,
                  alpha: float = ALPHA) -> list:
    """
    Run all five ECDF-based statistical tests in sequence.
    Returns list of 5 TestResult objects: [KS, Kuiper, CVM, AD, EMD]
    """
    return [
        ks_test(x_hist, x_new, alpha),
        kuiper_test(x_hist, x_new, alpha),
        cvm_test(x_hist, x_new, alpha),
        ad_test(x_hist, x_new, alpha),
        emd_test(x_hist, x_new, alpha),
    ]


if __name__ == "__main__":
    # Quick validation with two clearly different distributions
    rng = np.random.default_rng(42)
    x_h = rng.normal(5, 1, 50)
    x_n = rng.normal(8, 1, 50)   # Obvious drift

    print("=== Testing with drifted distributions (µ: 5 → 8) ===")
    for result in run_all_tests(x_h, x_n):
        print(result)

    print("\n=== Testing with similar distributions (µ: 5, 5) ===")
    x_same = rng.normal(5, 1, 50)
    for result in run_all_tests(x_h, x_same):
        print(result)
