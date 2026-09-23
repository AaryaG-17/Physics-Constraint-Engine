"""
test_sampler.py — Unit Tests for sampler.py

Coverage:
    - _sample_split_normal distribution shape and statistics
    - run_monte_carlo output shapes, dtypes, reproducibility
    - Physical invariants across all samples
    - Scalar vs vectorised consistency
    - Asymmetric uncertainty handling
    - Edge cases and invalid input rejection
"""

import sys
import math
import pytest
import numpy as np

sys.path.insert(0, "src")

from astropy import units as u
from pce.sampler import run_monte_carlo, _sample_split_normal
from pce.schemas import BoundsDistribution
from utils.constants import MIN_TRANSIT_DEPTH
from tests.fixtures.mock_data import (
    STELLAR_PARAMS_SOLAR,
    STELLAR_PARAMS_MDWARF,
    STELLAR_PARAMS_ASYMMETRIC,
    make_stellar_params,
)


# ---------------------------------------------------------------------------
# 1. _sample_split_normal
# ---------------------------------------------------------------------------

class TestSampleSplitNormal:

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        samples = _sample_split_normal(1.0, 0.1, 0.1, 10000, rng)
        assert samples.shape == (10000,)

    def test_mean_close_to_central_value(self):
        """With symmetric errors, sample mean should be close to central value."""
        rng = np.random.default_rng(0)
        samples = _sample_split_normal(5.0, 0.5, 0.5, 100_000, rng)
        assert samples.mean() == pytest.approx(5.0, abs=0.02)

    def test_symmetric_std_matches_sigma(self):
        """Symmetric errors → sample std should match sigma closely."""
        rng = np.random.default_rng(0)
        sigma = 0.3
        samples = _sample_split_normal(1.0, sigma, sigma, 100_000, rng)
        assert samples.std() == pytest.approx(sigma, rel=0.05)

    def test_asymmetric_upper_tail_wider(self):
        """When sigma_hi > sigma_lo, more spread above mean than below."""
        rng = np.random.default_rng(0)
        samples = _sample_split_normal(1.0, 0.1, 0.5, 100_000, rng)
        below = samples[samples < 1.0]
        above = samples[samples >= 1.0]
        assert above.std() > below.std()

    def test_asymmetric_lower_tail_wider(self):
        """When sigma_lo > sigma_hi, more spread below mean than above."""
        rng = np.random.default_rng(0)
        samples = _sample_split_normal(1.0, 0.5, 0.1, 100_000, rng)
        below = samples[samples < 1.0]
        above = samples[samples >= 1.0]
        assert below.std() > above.std()

    def test_zero_sigma_returns_constant(self):
        """Zero uncertainties → all samples equal to mean."""
        rng = np.random.default_rng(0)
        samples = _sample_split_normal(3.14, 0.0, 0.0, 1000, rng)
        assert np.allclose(samples, 3.14)

    def test_reproducible_with_same_seed(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        s1 = _sample_split_normal(1.0, 0.1, 0.2, 1000, rng1)
        s2 = _sample_split_normal(1.0, 0.1, 0.2, 1000, rng2)
        assert np.array_equal(s1, s2)

    def test_different_seeds_differ(self):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        s1 = _sample_split_normal(1.0, 0.1, 0.1, 1000, rng1)
        s2 = _sample_split_normal(1.0, 0.1, 0.1, 1000, rng2)
        assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# 2. run_monte_carlo — output structure
# ---------------------------------------------------------------------------

class TestRunMonteCarloOutput:

    @pytest.fixture
    def dist(self):
        return run_monte_carlo(
            STELLAR_PARAMS_SOLAR,
            observation_baseline_days=27.0,
            n_samples=10_000,
            seed=42,
        )

    def test_returns_bounds_distribution(self, dist):
        assert isinstance(dist, BoundsDistribution)

    def test_n_samples_recorded(self, dist):
        assert dist.n_samples == 10_000

    def test_seed_recorded(self, dist):
        assert dist.seed == 42

    def test_all_arrays_correct_shape(self, dist):
        for arr_name in [
            "period_min_samples", "period_max_samples",
            "duration_min_samples", "duration_max_samples",
            "depth_min_samples", "depth_max_samples",
        ]:
            arr = getattr(dist, arr_name)
            assert arr.shape == (10_000,), f"{arr_name} shape wrong: {arr.shape}"

    def test_all_arrays_float64(self, dist):
        for arr_name in [
            "period_min_samples", "period_max_samples",
            "duration_min_samples", "duration_max_samples",
            "depth_min_samples", "depth_max_samples",
        ]:
            arr = getattr(dist, arr_name)
            assert arr.dtype == np.float64, f"{arr_name} dtype wrong: {arr.dtype}"

    def test_no_nan_in_any_array(self, dist):
        for arr_name in [
            "period_min_samples", "period_max_samples",
            "duration_min_samples", "duration_max_samples",
            "depth_min_samples", "depth_max_samples",
        ]:
            arr = getattr(dist, arr_name)
            assert not np.any(np.isnan(arr)), f"{arr_name} contains NaN"

    def test_no_inf_in_any_array(self, dist):
        for arr_name in [
            "period_min_samples", "period_max_samples",
            "duration_min_samples", "duration_max_samples",
            "depth_min_samples", "depth_max_samples",
        ]:
            arr = getattr(dist, arr_name)
            assert not np.any(np.isinf(arr)), f"{arr_name} contains Inf"


# ---------------------------------------------------------------------------
# 3. Physical invariants
# ---------------------------------------------------------------------------

class TestPhysicalInvariants:

    @pytest.fixture
    def dist(self):
        return run_monte_carlo(
            STELLAR_PARAMS_SOLAR,
            observation_baseline_days=27.0,
            n_samples=10_000,
            seed=42,
        )

    def test_period_min_always_positive(self, dist):
        assert np.all(dist.period_min_samples > 0)

    def test_period_max_always_positive(self, dist):
        assert np.all(dist.period_max_samples > 0)

    def test_period_min_less_than_period_max(self, dist):
        assert np.all(dist.period_min_samples < dist.period_max_samples)

    def test_duration_min_always_positive(self, dist):
        assert np.all(dist.duration_min_samples > 0)

    def test_duration_max_always_positive(self, dist):
        assert np.all(dist.duration_max_samples > 0)

    def test_duration_min_less_than_duration_max(self, dist):
        assert np.all(dist.duration_min_samples < dist.duration_max_samples)

    def test_depth_min_always_positive(self, dist):
        assert np.all(dist.depth_min_samples > 0)

    def test_depth_max_less_than_one(self, dist):
        assert np.all(dist.depth_max_samples < 1.0)

    def test_depth_min_less_than_depth_max(self, dist):
        assert np.all(dist.depth_min_samples < dist.depth_max_samples)

    def test_period_max_constant(self):
        """period_max doesn't depend on stellar params — should be constant."""
        dist = run_monte_carlo(
            STELLAR_PARAMS_SOLAR,
            observation_baseline_days=27.0,
            n_samples=1000,
            seed=42,
        )
        assert np.allclose(dist.period_max_samples, dist.period_max_samples[0])

    def test_period_max_value_27d_2transits(self):
        """27-day baseline, 2 transits → period_max = 13.5 days."""
        dist = run_monte_carlo(
            STELLAR_PARAMS_SOLAR,
            observation_baseline_days=27.0,
            min_transits=2,
            n_samples=100,
            seed=42,
        )
        assert dist.period_max_samples[0] == pytest.approx(13.5, rel=1e-6)

    def test_period_max_value_27d_3transits(self):
        """27-day baseline, 3 transits → period_max = 9.0 days."""
        dist = run_monte_carlo(
            STELLAR_PARAMS_SOLAR,
            observation_baseline_days=27.0,
            min_transits=3,
            n_samples=100,
            seed=42,
        )
        assert dist.period_max_samples[0] == pytest.approx(9.0, rel=1e-6)

    def test_depth_min_constant_equals_floor(self):
        """depth_min should equal min_transit_depth for all samples."""
        floor = 200e-6
        dist = run_monte_carlo(
            STELLAR_PARAMS_SOLAR,
            observation_baseline_days=27.0,
            min_transit_depth=floor,
            n_samples=1000,
            seed=42,
        )
        assert np.allclose(dist.depth_min_samples, floor)

    def test_period_min_range_physically_plausible(self, dist):
            """Roche period should be between ~0.1 and ~3.1 days for typical stars.
            
            With corrected planet density range (0.03–14.1 g/cm³):
            - Lower bound 0.03 g/cm³ (super-puff floor) → P_Roche ≈ 3.03 days
            - Upper bound 14.1 g/cm³ (TOI-4603b) → P_Roche ≈ 0.33 days
            """
            assert dist.period_min_samples.min() > 0.1
            assert dist.period_min_samples.max() < 3.1

    def test_duration_max_range_hours(self, dist):
        """Max transit duration for a TESS-like baseline should be 1-20 hrs."""
        assert dist.duration_max_samples.min() > 0.5
        assert dist.duration_max_samples.max() < 20.0


# ---------------------------------------------------------------------------
# 4. Reproducibility and seed behaviour
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_identical_output(self):
        d1 = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, seed=42, n_samples=1000)
        d2 = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, seed=42, n_samples=1000)
        assert np.array_equal(d1.period_min_samples, d2.period_min_samples)
        assert np.array_equal(d1.duration_max_samples, d2.duration_max_samples)

    def test_different_seed_different_output(self):
        d1 = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, seed=1, n_samples=1000)
        d2 = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, seed=2, n_samples=1000)

        # Deterministic physical assumptions must remain identical.
        assert np.array_equal(d1.period_min_samples, d2.period_min_samples)

        # Stellar-uncertainty propagation must still depend on the seed.
        assert not np.array_equal(d1.duration_min_samples, d2.duration_min_samples)
        assert not np.array_equal(d1.duration_max_samples, d2.duration_max_samples)
        assert not np.array_equal(d1.depth_max_samples, d2.depth_max_samples)

    def test_n_samples_respected(self):
        for n in [100, 500, 10_000]:
            dist = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, n_samples=n, seed=0)
            assert dist.n_samples == n
            assert dist.period_min_samples.shape == (n,)


# ---------------------------------------------------------------------------
# 5. Different stellar types
# ---------------------------------------------------------------------------

class TestDifferentStellarTypes:

    def test_mdwarf_runs_without_error(self):
        dist = run_monte_carlo(STELLAR_PARAMS_MDWARF, 27.0, n_samples=1000, seed=42)
        assert isinstance(dist, BoundsDistribution)

    def test_mdwarf_physical_invariants(self):
        dist = run_monte_carlo(STELLAR_PARAMS_MDWARF, 27.0, n_samples=1000, seed=42)
        assert np.all(dist.period_min_samples < dist.period_max_samples)
        assert np.all(dist.duration_min_samples < dist.duration_max_samples)
        assert np.all(dist.depth_min_samples < dist.depth_max_samples)

    def test_asymmetric_star_runs_without_error(self):
        dist = run_monte_carlo(STELLAR_PARAMS_ASYMMETRIC, 27.0, n_samples=1000, seed=42)
        assert isinstance(dist, BoundsDistribution)

    def test_larger_star_longer_max_duration(self):
        """Larger stellar radius → longer transit duration."""
        dist_solar  = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, n_samples=5000, seed=42)
        big_star = make_stellar_params(radius=3.0, radius_err_lo=0.1, radius_err_hi=0.1,
                                        mass=2.0, mass_err_lo=0.1, mass_err_hi=0.1)
        dist_big = run_monte_carlo(big_star, 27.0, n_samples=5000, seed=42)
        assert dist_big.duration_max_samples.mean() > dist_solar.duration_max_samples.mean()

    def test_longer_baseline_larger_pmax(self):
        """Longer observation → larger period_max."""
        dist_short = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0,  n_samples=100, seed=42)
        dist_long  = run_monte_carlo(STELLAR_PARAMS_SOLAR, 365.0, n_samples=100, seed=42)
        assert dist_long.period_max_samples[0] > dist_short.period_max_samples[0]

    def test_zero_uncertainty_star_constant_samples(self):
        """A star with zero uncertainties → all stellar samples are identical."""
        zero_err_star = make_stellar_params(
            mass_err_lo=0.0, mass_err_hi=0.0,
            radius_err_lo=0.0, radius_err_hi=0.0,
            teff_err_lo=0.0, teff_err_hi=0.0,
            luminosity_err_lo=0.0, luminosity_err_hi=0.0,
        )
        dist = run_monte_carlo(zero_err_star, 27.0, n_samples=1000, seed=42)
        # duration_max should be nearly constant (only planet radius varies)
        assert dist.duration_max_samples.std() < dist.duration_max_samples.mean() * 0.5


# ---------------------------------------------------------------------------
# 6. Invalid inputs
# ---------------------------------------------------------------------------

class TestInvalidInputs:

    def test_zero_baseline_raises(self):
        with pytest.raises(ValueError, match="observation_baseline_days"):
            run_monte_carlo(STELLAR_PARAMS_SOLAR, 0.0)

    def test_negative_baseline_raises(self):
        with pytest.raises(ValueError, match="observation_baseline_days"):
            run_monte_carlo(STELLAR_PARAMS_SOLAR, -10.0)

    def test_min_transits_less_than_2_raises(self):
        with pytest.raises(ValueError, match="min_transits"):
            run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, min_transits=1)

    def test_zero_depth_floor_raises(self):
        with pytest.raises(ValueError, match="min_transit_depth"):
            run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, min_transit_depth=0.0)

    def test_negative_depth_floor_raises(self):
        with pytest.raises(ValueError, match="min_transit_depth"):
            run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, min_transit_depth=-1e-6)

    def test_zero_n_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, n_samples=0)
