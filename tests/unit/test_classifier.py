"""
test_classifier.py — Unit Tests for classifier.py

Coverage:
    - Zone 1/2/3 bounds computed from known distributions
    - Zone 1 ⊆ Zone 2 (Zone 1 is always tighter)
    - Percentile accuracy against numpy ground truth
    - Edge cases: identical samples, extreme spread
    - Invalid percentile inputs rejected
"""

import sys
import pytest
import numpy as np

sys.path.insert(0, "src")

from pce.schemas import BoundsDistribution, ZoneBounds
from pce.classifier import classify_zones
from pce.sampler import run_monte_carlo
from tests.fixtures.mock_data import STELLAR_PARAMS_SOLAR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dist(
    period_min=None, period_max=None,
    duration_min=None, duration_max=None,
    depth_min=None, depth_max=None,
    n=10_000, seed=42,
) -> BoundsDistribution:
    """Build a BoundsDistribution from explicit arrays or defaults."""
    rng = np.random.default_rng(seed)
    n = int(n)
    return BoundsDistribution(
        n_samples=n,
        seed=seed,
        period_min_samples   = period_min   if period_min   is not None else rng.uniform(0.2, 0.8, n),
        period_max_samples   = period_max   if period_max   is not None else np.full(n, 13.5),
        duration_min_samples = duration_min if duration_min is not None else rng.uniform(0.3, 0.7, n),
        duration_max_samples = duration_max if duration_max is not None else rng.uniform(3.0, 6.0, n),
        depth_min_samples    = depth_min    if depth_min    is not None else np.full(n, 200e-6),
        depth_max_samples    = depth_max    if depth_max    is not None else rng.uniform(0.001, 0.05, n),
    )


# ---------------------------------------------------------------------------
# 1. Return type and structure
# ---------------------------------------------------------------------------

class TestClassifyZonesReturnType:

    def test_returns_three_zone_bounds(self):
        dist = _make_dist()
        result = classify_zones(dist)
        assert len(result) == 3

    def test_all_zones_are_zone_bounds(self):
        dist = _make_dist()
        z1, z2, z3 = classify_zones(dist)
        assert isinstance(z1, ZoneBounds)
        assert isinstance(z2, ZoneBounds)
        assert isinstance(z3, ZoneBounds)

    def test_zone1_fields_not_none(self):
        dist = _make_dist()
        z1, _, _ = classify_zones(dist)
        assert z1.period_min is not None
        assert z1.period_max is not None
        assert z1.duration_min is not None
        assert z1.duration_max is not None
        assert z1.depth_min is not None
        assert z1.depth_max is not None

    def test_zone2_fields_not_none(self):
        dist = _make_dist()
        _, z2, _ = classify_zones(dist)
        assert z2.period_min is not None
        assert z2.period_max is not None


# ---------------------------------------------------------------------------
# 2. Zone 1 ⊆ Zone 2 — Zone 1 must always be tighter
# ---------------------------------------------------------------------------

class TestZoneOrdering:

    @pytest.fixture
    def zones(self):
        dist = _make_dist()
        return classify_zones(dist)

    def test_zone1_period_min_ge_zone2_period_min(self, zones):
        z1, z2, _ = zones
        assert z1.period_min >= z2.period_min

    def test_zone1_period_max_le_zone2_period_max(self, zones):
        z1, z2, _ = zones
        assert z1.period_max <= z2.period_max

    def test_zone1_duration_min_ge_zone2_duration_min(self, zones):
        z1, z2, _ = zones
        assert z1.duration_min >= z2.duration_min

    def test_zone1_duration_max_le_zone2_duration_max(self, zones):
        z1, z2, _ = zones
        assert z1.duration_max <= z2.duration_max

    def test_zone1_depth_min_ge_zone2_depth_min(self, zones):
        z1, z2, _ = zones
        assert z1.depth_min >= z2.depth_min

    def test_zone1_depth_max_ge_zone2_depth_max(self, zones):
        # depth_max uses HIGH percentile for both zones:
        # Zone 1 uses z1_hi (e.g. P95), Zone 2 uses z2_hi (e.g. P75).
        # Since z1_hi > z2_hi, Zone 1 depth_max >= Zone 2 depth_max.
        z1, z2, _ = zones
        assert z1.depth_max >= z2.depth_max

    def test_zone1_period_min_lt_zone1_period_max(self, zones):
        z1, _, _ = zones
        assert z1.period_min < z1.period_max

    def test_zone2_period_min_lt_zone2_period_max(self, zones):
        _, z2, _ = zones
        assert z2.period_min < z2.period_max


# ---------------------------------------------------------------------------
# 3. Percentile accuracy vs numpy ground truth
# ---------------------------------------------------------------------------

class TestPercentileAccuracy:

    def test_zone1_period_min_matches_numpy_p95(self):
        """
        Zone 1 period_min bound = P95 of period_min_samples
        (90% confidence → z1_hi = (1+0.90)/2 * 100 = 95.0)
        """
        n = 100_000
        arr = np.random.default_rng(0).uniform(0.2, 0.8, n)
        dist = _make_dist(period_min=arr, n=n, seed=0)
        z1, _, _ = classify_zones(dist, zone1_percentile=0.90)
        expected = float(np.percentile(arr, 95.0))
        assert z1.period_min == pytest.approx(expected, rel=1e-9)

    def test_zone1_period_max_matches_numpy_p5(self):
        """
        Zone 1 period_max bound = P5 of period_max_samples
        (90% confidence → z1_lo = (1-0.90)/2 * 100 = 5.0)
        """
        n = 100_000
        arr = np.full(n, 13.5)
        dist = _make_dist(period_max=arr, n=n, seed=0)
        z1, _, _ = classify_zones(dist, zone1_percentile=0.90)
        assert z1.period_max == pytest.approx(13.5, rel=1e-9)

    def test_zone2_duration_min_matches_numpy_p75(self):
        """
        Zone 2 duration_min bound = P75 of duration_min_samples
        (50% confidence → z2_hi = (1+0.50)/2 * 100 = 75.0)
        """
        n = 100_000
        arr = np.random.default_rng(1).uniform(0.3, 0.7, n)
        dist = _make_dist(duration_min=arr, n=n, seed=1)
        _, z2, _ = classify_zones(dist, zone2_percentile=0.50)
        expected = float(np.percentile(arr, 75.0))
        assert z2.duration_min == pytest.approx(expected, rel=1e-9)

    def test_zone2_depth_max_matches_numpy_p75(self):
        """
        Zone 2 depth_max uses the HIGH percentile of depth_max_samples,
        so large planets (gas giants) are inside the zone.
        zone2_percentile=0.50 → z2_hi = (1+0.50)/2 * 100 = 75.0
        """
        n = 100_000
        arr = np.random.default_rng(2).uniform(0.001, 0.05, n)
        dist = _make_dist(depth_max=arr, n=n, seed=2)
        _, z2, _ = classify_zones(dist, zone2_percentile=0.50)
        expected = float(np.percentile(arr, 75.0))
        assert z2.depth_max == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_samples_identical_zone1_equals_zone2(self):
        """All samples the same → Zone 1 and Zone 2 bounds are identical."""
        n = 1000
        const = 5.0
        dist = _make_dist(
            period_min=np.full(n, const),
            period_max=np.full(n, 13.5),
            duration_min=np.full(n, 1.0),
            duration_max=np.full(n, 4.0),
            depth_min=np.full(n, 200e-6),
            depth_max=np.full(n, 0.01),
            n=n,
        )
        z1, z2, _ = classify_zones(dist)
        assert z1.period_min == pytest.approx(z2.period_min, rel=1e-9)

    def test_wide_spread_zone2_wider_than_zone1(self):
        """High-variance distribution → Zone 2 noticeably wider than Zone 1."""
        n = 10_000
        arr = np.random.default_rng(99).uniform(0.1, 10.0, n)
        dist = _make_dist(period_min=arr, n=n, seed=99)
        z1, z2, _ = classify_zones(dist)
        assert (z2.period_max - z2.period_min) >= (z1.period_max - z1.period_min)

    def test_zone3_period_max_equals_zone2_period_min(self):
        """Zone 3 upper bound should equal Zone 2 lower bound."""
        dist = _make_dist()
        _, z2, z3 = classify_zones(dist)
        assert z3.period_max == pytest.approx(z2.period_min, rel=1e-9)

    def test_zone3_period_min_is_none(self):
        """Zone 3 has no lower period bound (open-ended below)."""
        dist = _make_dist()
        _, _, z3 = classify_zones(dist)
        assert z3.period_min is None

    def test_single_sample(self):
        """Single sample should not crash — zones collapse to one point."""
        dist = _make_dist(
            period_min=np.array([0.5]),
            period_max=np.array([13.5]),
            duration_min=np.array([1.0]),
            duration_max=np.array([4.0]),
            depth_min=np.array([200e-6]),
            depth_max=np.array([0.01]),
            n=1,
        )
        z1, z2, z3 = classify_zones(dist)
        assert z1 is not None
        assert z2 is not None


# ---------------------------------------------------------------------------
# 5. Custom percentile thresholds
# ---------------------------------------------------------------------------

class TestCustomPercentiles:

    def test_higher_percentile_raises_period_min_bound(self):
        """
        Higher zone1_percentile uses a higher percentile for period_min bound.
        90% -> P95(period_min); 80% -> P90(period_min). P95 >= P90 always.
        """
        rng = np.random.default_rng(7)
        n = 50_000
        arr = rng.uniform(0.2, 0.8, n)
        dist = _make_dist(period_min=arr, n=n, seed=7)
        z1_90, _, _ = classify_zones(dist, zone1_percentile=0.90, zone2_percentile=0.50)
        z1_80, _, _ = classify_zones(dist, zone1_percentile=0.80, zone2_percentile=0.40)
        assert z1_90.period_min >= z1_80.period_min

    def test_inverted_percentiles_raises(self):
        dist = _make_dist()
        with pytest.raises(ValueError, match="zone2_percentile < zone1_percentile"):
            classify_zones(dist, zone1_percentile=0.50, zone2_percentile=0.90)

    def test_zero_zone1_percentile_raises(self):
        dist = _make_dist()
        with pytest.raises(ValueError):
            classify_zones(dist, zone1_percentile=0.0, zone2_percentile=0.0)

    def test_one_zone1_percentile_raises(self):
        dist = _make_dist()
        with pytest.raises(ValueError):
            classify_zones(dist, zone1_percentile=1.0, zone2_percentile=0.5)

    def test_equal_percentiles_raises(self):
        dist = _make_dist()
        with pytest.raises(ValueError):
            classify_zones(dist, zone1_percentile=0.75, zone2_percentile=0.75)


# ---------------------------------------------------------------------------
# 6. Integration with real sampler output
# ---------------------------------------------------------------------------

class TestWithRealSamplerOutput:

    @pytest.fixture
    def dist(self):
        return run_monte_carlo(
            STELLAR_PARAMS_SOLAR,
            observation_baseline_days=27.0,
            n_samples=10_000,
            seed=42,
        )

    def test_classify_real_dist_runs(self, dist):
        z1, z2, z3 = classify_zones(dist)
        assert isinstance(z1, ZoneBounds)

    def test_zone1_period_range_sensible(self, dist):
        """Zone 1 period range should be within 0–14 days for 27d baseline."""
        z1, _, _ = classify_zones(dist)
        assert 0 < z1.period_min < z1.period_max <= 13.5

    def test_zone1_duration_range_sensible(self, dist):
        """Zone 1 duration should be in hours, positive, ordered."""
        z1, _, _ = classify_zones(dist)
        assert 0 < z1.duration_min < z1.duration_max

    def test_zone1_depth_range_sensible(self, dist):
        """Zone 1 depth should be between 0 and 1."""
        z1, _, _ = classify_zones(dist)
        assert 0 < z1.depth_min < z1.depth_max < 1.0
