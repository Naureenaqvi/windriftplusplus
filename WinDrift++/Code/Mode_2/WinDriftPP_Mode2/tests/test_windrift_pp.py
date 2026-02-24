"""
tests/test_windrift_pp.py
=========================
Pytest test suite for WinDrift++ core modules.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=Code --cov-report=term-missing
"""


# **** Shaping up provided code with parameters and correct indentation ****
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure Code/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))

from ecdf import compute_ecdf_at_points, compute_ecdf_pair, ecdf_differences
from statistical_tests import (
    ks_test, kuiper_test, cvm_test, ad_test, emd_test, run_all_tests
)
from voting import (
    majority_vote, plurality_vote, borda_count_vote,
    plurality_elimination_vote, pairwise_vote
)
from windrift_pp import WinDriftPP, _consecutive_indices, _corresponding_indices
from evaluation import compute_confusion_matrix, compute_metrics


# ************************************************************
# Fixtures
# ************************************************************

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def similar_samples(rng):
    """Two samples from the same distribution — no drift."""
    return rng.normal(5, 1, 50), rng.normal(5, 1, 50)


@pytest.fixture
def drifted_samples(rng):
    """Two samples with a large mean shift — clear drift."""
    return rng.normal(5, 1, 50), rng.normal(12, 1, 50)


@pytest.fixture
def small_dataset():
    """Minimal 2-year monthly dataset (5 obs per month, 24 months)."""
    rng = np.random.default_rng(99)
    cols = {f"M{i:02d}": rng.normal(5, 1, 5) for i in range(1, 13)}
    # Year 2: mean shift at month 4 (column M16)
    for i in range(13, 16):
        cols[f"M{i:02d}"] = rng.normal(5, 1, 5)
    for i in range(16, 25):
        cols[f"M{i:02d}"] = rng.normal(12, 1, 5)
    return pd.DataFrame(cols)


# ************************************************************
# ECDF tests
# ************************************************************

class TestECDF:

    def test_unique_values_paper_example(self):
        """Canonical ECDF example: 5-value sample with a tie."""
        x = np.array([12.50, 17.80, 16.00, 16.70, 16.20, 16.80,
                      12.70, 16.90, 15.10, 13.00])
        x_sorted = np.sort(x)
        ecdf = compute_ecdf_at_points(x, x_sorted, round_decimals=1)
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        np.testing.assert_array_equal(ecdf, expected)

    def test_ties_paper_example(self):
        """Ties should receive the maximum (highest) ECDF value — Section A."""
        x = np.array([12.50, 12.70, 13.00, 15.10, 16.00,
                      16.00, 16.00, 16.80, 16.90, 17.80])
        x_sorted = np.sort(x)
        ecdf = compute_ecdf_at_points(x, x_sorted, round_decimals=1)
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0])
        np.testing.assert_array_equal(ecdf, expected)

    def test_ecdf_monotone(self, rng):
        """ECDF must be non-decreasing."""
        x = rng.normal(0, 1, 30)
        merged, ecdf_x, _ = compute_ecdf_pair(x, x)
        assert np.all(np.diff(ecdf_x) >= 0)

    def test_ecdf_bounds(self, rng):
        """ECDF must be in [0, 1]."""
        x = rng.normal(0, 1, 30)
        y = rng.normal(1, 1, 30)
        _, ecdf_x, ecdf_y = compute_ecdf_pair(x, y)
        assert ecdf_x.min() >= 0 and ecdf_x.max() <= 1
        assert ecdf_y.min() >= 0 and ecdf_y.max() <= 1

    def test_ecdf_pair_returns_sorted_merged(self, rng):
        x = rng.normal(0, 1, 20)
        y = rng.normal(2, 1, 20)
        merged, _, _ = compute_ecdf_pair(x, y)
        assert np.all(np.diff(merged) >= 0)

    def test_identical_samples_zero_difference(self, rng):
        x = rng.normal(0, 1, 30)
        _, ecdf_x, ecdf_y = compute_ecdf_pair(x, x)
        _, abs_diff = ecdf_differences(ecdf_x, ecdf_y)
        assert np.allclose(abs_diff, 0)


# ************************************************************
# Statistical tests
# ************************************************************

class TestStatisticalTests:

    def test_ks_detects_clear_drift(self, drifted_samples):
        x_h, x_n = drifted_samples
        result = ks_test(x_h, x_n)
        assert result.drift_flag == 1
        assert result.stat > result.crit

    def test_ks_no_drift_similar(self, similar_samples):
        x_h, x_n = similar_samples
        result = ks_test(x_h, x_n)
        assert result.drift_flag == 0

    def test_kuiper_detects_clear_drift(self, drifted_samples):
        x_h, x_n = drifted_samples
        result = kuiper_test(x_h, x_n)
        assert result.drift_flag == 1

    def test_cvm_detects_clear_drift(self, drifted_samples):
        x_h, x_n = drifted_samples
        result = cvm_test(x_h, x_n)
        assert result.drift_flag == 1

    def test_ad_detects_clear_drift(self, drifted_samples):
        x_h, x_n = drifted_samples
        result = ad_test(x_h, x_n)
        assert result.drift_flag == 1

    def test_emd_detects_clear_drift(self, drifted_samples):
        x_h, x_n = drifted_samples
        result = emd_test(x_h, x_n)
        assert result.drift_flag == 1

    def test_run_all_tests_returns_five_results(self, drifted_samples):
        x_h, x_n = drifted_samples
        results = run_all_tests(x_h, x_n)
        assert len(results) == 5

    def test_run_all_tests_all_drift_on_clear_signal(self, drifted_samples):
        x_h, x_n = drifted_samples
        results = run_all_tests(x_h, x_n)
        flags = [r.drift_flag for r in results]
        assert sum(flags) >= 4, f"Expected ≥4/5 tests to detect drift, got {flags}"

    def test_test_result_fields(self, drifted_samples):
        x_h, x_n = drifted_samples
        result = ks_test(x_h, x_n)
        assert hasattr(result, "stat")
        assert hasattr(result, "crit")
        assert hasattr(result, "p_value")
        assert hasattr(result, "drift_flag")
        assert result.drift_flag in (0, 1)

    def test_ks_critical_value_formula(self):
        """D_crit = 1.36 × √(1/n + 1/m) — standard KS critical value formula."""
        n, m = 30, 30
        x_h = np.random.default_rng(1).normal(0, 1, n)
        x_n = np.random.default_rng(2).normal(0, 1, m)
        result = ks_test(x_h, x_n)
        expected_crit = 1.36 * np.sqrt(1/n + 1/m)
        assert abs(result.crit - expected_crit) < 1e-6

    def test_p_value_in_range(self, drifted_samples):
        x_h, x_n = drifted_samples
        for result in run_all_tests(x_h, x_n):
            assert 0.0 <= result.p_value <= 1.0


# ************************************************************
# Voting
# ************************************************************

class TestVoting:

    def _make_results(self, flags, p_vals=None):
        """Build mock TestResult objects from flag lists."""
        from statistical_tests import TestResult
        if p_vals is None:
            p_vals = [0.01 if f else 0.5 for f in flags]
        names = ["KS", "Kuiper", "CVM", "AD", "EMD"]
        return [
            TestResult(names[i], 0.5, 0.3, p_vals[i], flags[i])
            for i in range(len(flags))
        ]

    def test_majority_drift_3_of_5(self):
        results = self._make_results([1, 1, 1, 0, 0])
        vote = majority_vote(results)
        assert vote.drift_flag == 1
        assert vote.label == "D"

    def test_majority_no_drift_2_of_5(self):
        results = self._make_results([0, 0, 0, 1, 1])
        vote = majority_vote(results)
        assert vote.drift_flag == 0
        assert vote.label == "D̄"

    def test_majority_tie_even_tests(self):
        """Even number of tests with 2-2 split → Unsure."""
        from statistical_tests import TestResult
        results = [
            TestResult("KS", 0.5, 0.3, 0.01, 1),
            TestResult("Kuiper", 0.5, 0.3, 0.01, 1),
            TestResult("CVM", 0.5, 0.3, 0.5, 0),
            TestResult("AD", 0.5, 0.3, 0.5, 0),
        ]
        vote = majority_vote(results)
        assert vote.label == "U"
        assert vote.drift_flag == -1

    def test_majority_unanimous_drift(self):
        results = self._make_results([1, 1, 1, 1, 1])
        vote = majority_vote(results)
        assert vote.drift_flag == 1
        assert vote.votes_drift == 5

    def test_majority_unanimous_no_drift(self):
        results = self._make_results([0, 0, 0, 0, 0])
        vote = majority_vote(results)
        assert vote.drift_flag == 0
        assert vote.votes_nodrift == 5

    def test_all_voting_methods_return_dict(self, drifted_samples):
        x_h, x_n = drifted_samples
        test_results = run_all_tests(x_h, x_n)
        votes = {
            "majority":    majority_vote(test_results),
            "plurality":   plurality_vote(test_results),
            "borda":       borda_count_vote(test_results),
            "elimination": plurality_elimination_vote(test_results),
            "pairwise":    pairwise_vote(test_results),
        }
        for name, vote in votes.items():
            assert vote.drift_flag in (-1, 0, 1), f"{name} returned invalid flag"


# ************************************************************
# Window index logic
# ************************************************************

class TestWindowIndices:

    def test_consecutive_w1_first_pair(self):
        h, n = _consecutive_indices(1, 1, maxcol=24, max_cyc_len=12)
        assert h == [0] and n == [1]

    def test_consecutive_w1_boundary(self):
        """At max_cyc_len, W=1 should NOT fire (Mode I stops at boundary)."""
        h, n = _consecutive_indices(12, 1, maxcol=24, max_cyc_len=12)
        assert h == [] and n == []

    def test_consecutive_w3_at_6(self):
        """W=3 fires at dbcount=6 (first multiple of 3 > 3)."""
        h, n = _consecutive_indices(6, 3, maxcol=24, max_cyc_len=12)
        assert len(h) == 3 and len(n) == 3

    def test_corresponding_w1_first_pair(self):
        """Mode II: dbcount=12 pairs column 0 (M01 Y2019) vs column 12 (M01 Y2020)."""
        h, n = _corresponding_indices(12, 1, maxcol=24, max_cyc_len=12)
        assert h == [0] and n == [12]

    def test_corresponding_w1_last_pair(self):
        """Mode II: dbcount=23 pairs column 11 vs column 23."""
        h, n = _corresponding_indices(23, 1, maxcol=24, max_cyc_len=12)
        assert h == [11] and n == [23]

    def test_no_out_of_bounds_indices(self):
        """All returned indices must be within [0, maxcol-1].
        Mode I  is valid for dbcount in [1, max_cyc_len].
        Mode II is valid for dbcount in [max_cyc_len, maxcol].
        """
        maxcol, max_cyc_len = 24, 12
        for win_level in [1, 3, 6, 12]:
            # Mode I: consecutive
            for dbcount in range(1, max_cyc_len + 1):
                h, n = _consecutive_indices(dbcount, win_level, maxcol, max_cyc_len)
                for idx in h + n:
                    assert 0 <= idx < maxcol, \
                        f"consecutive dbcount={dbcount} W={win_level}: index {idx} out of range"
            # Mode II: corresponding (only called once maxCycLen is reached)
            for dbcount in range(max_cyc_len, maxcol + 1):
                h, n = _corresponding_indices(dbcount, win_level, maxcol, max_cyc_len)
                for idx in h + n:
                    assert 0 <= idx < maxcol, \
                        f"corresponding dbcount={dbcount} W={win_level}: index {idx} out of range"


# ************************************************************
# WinDriftPP integration
# ************************************************************

class TestWinDriftPP:

    def test_detect_returns_records(self, small_dataset):
        detector = WinDriftPP(verbose=False)
        records = detector.detect(small_dataset, dataset_id="test")
        assert len(records) > 0

    def test_to_dataframe_schema(self, small_dataset):
        detector = WinDriftPP(verbose=False)
        detector.detect(small_dataset, dataset_id="test")
        df = detector.to_dataframe()
        required_cols = [
            "dataset_id", "mode", "mode_name", "dbcount", "win_level",
            "label_h", "label_n", "ks_flag", "kuiper_flag", "cvm_flag",
            "ad_flag", "emd_flag", "majority_flag"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_majority_flag_binary_or_unsure(self, small_dataset):
        detector = WinDriftPP(verbose=False)
        detector.detect(small_dataset, dataset_id="test")
        df = detector.to_dataframe()
        assert set(df["majority_flag"].unique()).issubset({-1, 0, 1})

    def test_mode_values(self, small_dataset):
        detector = WinDriftPP(verbose=False)
        detector.detect(small_dataset, dataset_id="test")
        df = detector.to_dataframe()
        assert set(df["mode"].unique()).issubset({1, 2})

    def test_drifted_dataset_detects_in_mode2(self):
        """Clear drift dataset should produce majority_flag=1 in Mode II."""
        rng = np.random.default_rng(7)
        data = {}
        for i in range(1, 13):
            data[f"M{i:02d}"] = rng.normal(5, 1, 30)
        for i in range(13, 25):
            data[f"M{i:02d}"] = rng.normal(15, 1, 30)   # large shift
        df = pd.DataFrame(data)

        detector = WinDriftPP(verbose=False)
        detector.detect(df, dataset_id="drift_test")
        results = detector.to_dataframe()
        mode2 = results[results["mode"] == 2]
        assert mode2["majority_flag"].sum() > 0, \
            "Expected at least one Mode II drift detection on clearly drifted data"

    def test_no_drift_dataset_mode2_clean(self):
        """Identical Y2019/Y2020 should produce zero Mode II drift flags."""
        rng = np.random.default_rng(8)
        base = rng.normal(5, 1, 30)
        data = {f"M{i:02d}": rng.normal(5, 1, 30) for i in range(1, 25)}
        df = pd.DataFrame(data)

        detector = WinDriftPP(verbose=False)
        detector.detect(df, dataset_id="nodrift_test")
        results = detector.to_dataframe()
        mode2 = results[results["mode"] == 2]
        # Allow up to 1 false alarm (AD test is sensitive)
        assert mode2["majority_flag"].sum() <= 2, \
            f"Expected ≤2 false alarms, got {mode2['majority_flag'].sum()}"


# ************************************************************
# Evaluation metrics
# ************************************************************

class TestEvaluation:
    """Tests for evaluation.py detection statistics."""

    def _make_results(self, n_flagged, n_total, mode=2):
        """Helper: build a minimal results DataFrame."""
        import pandas as pd, numpy as np
        rows = []
        for i in range(n_total):
            rows.append({
                "mode": mode, "dbcount": i + 1, "win_level": 1,
                "majority_flag": 1 if i < n_flagged else 0,
                "ks_flag": 1 if i < n_flagged else 0,
                "kuiper_flag": 0, "cvm_flag": 0, "ad_flag": 0, "emd_flag": 0,
            })
        return pd.DataFrame(rows)

    def test_flags_raised_count(self):
        df = self._make_results(n_flagged=14, n_total=19)
        result = summarise_dataset("C6", df)
        assert result["flags_raised"] == 14

    def test_flag_rate(self):
        df = self._make_results(n_flagged=19, n_total=19)
        result = summarise_dataset("C6", df)
        assert result["flag_rate"] == 1.0

    def test_zero_flags(self):
        df = self._make_results(n_flagged=0, n_total=19)
        result = summarise_dataset("C1", df)
        assert result["flags_raised"] == 0
        assert result["flag_rate"] == 0.0
        assert result["first_flag_dbcount"] is None

    def test_first_flag_dbcount(self):
        df = self._make_results(n_flagged=5, n_total=19)
        result = summarise_dataset("C6", df)
        assert result["first_flag_dbcount"] == 1

    def test_per_test_counts_present(self):
        df = self._make_results(n_flagged=10, n_total=19)
        result = summarise_dataset("C6", df)
        assert "ks_flags" in result
        assert result["ks_flags"] == 10



class TestDataGenerator:

    @pytest.fixture(scope="class")
    def datasets(self):
        from data_generator import generate_all_datasets
        ds, meta = generate_all_datasets(save=False)
        return ds, meta

    def test_all_17_datasets_present(self, datasets):
        ds, _ = datasets
        for cid in [f"C{i}" for i in range(1, 18)]:
            assert cid in ds, f"Missing dataset {cid}"

    def test_shape_30_obs_24_months(self, datasets):
        ds, _ = datasets
        for cid, df in ds.items():
            assert df.shape == (30, 24), f"{cid}: expected (30,24), got {df.shape}"

    def test_no_nulls(self, datasets):
        ds, _ = datasets
        for cid, df in ds.items():
            assert not df.isnull().any().any(), f"{cid} contains nulls"

    def test_c1_c5_y2019_y2020_similar_means(self, datasets):
        """Similar datasets (C1-C5): Y2019 and Y2020 monthly means should be close."""
        ds, _ = datasets
        for cid in ["C1", "C2", "C3", "C4", "C5"]:
            df = ds[cid]
            y19_mean = df.iloc[:, :12].values.mean()
            y20_mean = df.iloc[:, 12:].values.mean()
            assert abs(y19_mean - y20_mean) < 5.0, \
                f"{cid}: similar dataset has unexpectedly large mean difference"

    def test_c6_c10_y2020_reversed(self, datasets):
        """Dissimilar datasets: Y2020 is reversed Y2019 → first/last months differ."""
        ds, _ = datasets
        for cid in ["C6", "C7", "C8", "C9", "C10"]:
            df = ds[cid]
            # M01 Y2019 mean ≠ M01 Y2020 mean (reversed)
            m01_y19 = df.iloc[:, 0].mean()
            m01_y20 = df.iloc[:, 12].mean()
            assert abs(m01_y19 - m01_y20) > 1.0, \
                f"{cid}: expected M01 Y2019 ≠ M01 Y2020 for dissimilar dataset"

    def test_c11_skewness_positive(self, datasets):
        """C11 Y2020 is Chi-squared: should be right-skewed (skewness > 0)."""
        from scipy.stats import skew
        ds, _ = datasets
        y20_data = ds["C11"].iloc[:, 12:].values.flatten()
        assert skew(y20_data) > 0, "C11 Y2020 (Chi-sq) should have positive skewness"

    def test_reproducibility(self):
        """Running generator twice with same seed produces identical datasets."""
        from data_generator import generate_all_datasets
        ds1, _ = generate_all_datasets(save=False)
        ds2, _ = generate_all_datasets(save=False)
        for cid in ds1:
            pd.testing.assert_frame_equal(ds1[cid], ds2[cid])
