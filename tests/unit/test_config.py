"""
test_config.py — Unit Tests for config.py

Coverage:
    - build_tls_config returns valid ZoneMap
    - All metadata fields present and correct
    - Zone bounds pass through unchanged
    - ZoneMap fields accessible for TLS usage pattern
"""

import sys
import pytest
from datetime import datetime, timezone

sys.path.insert(0, "src")

from pce.schemas import ZoneBounds, ZoneMap
from pce.config import build_tls_config
from pce.classifier import classify_zones
from pce.sampler import run_monte_carlo
from tests.fixtures.mock_data import STELLAR_PARAMS_SOLAR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zones():
    dist = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, n_samples=1000, seed=42)
    return classify_zones(dist)


def _default_zone_map(**overrides) -> ZoneMap:
    z1, z2, z3 = _make_zones()
    kwargs = dict(
        zone1=z1, zone2=z2, zone3=z3,
        star_id="TIC_25155310",
        n_samples=1000,
        seed=42,
        observation_baseline_days=27.0,
        min_transits=2,
        min_transit_depth=200e-6,
        catalog_source="TIC-8",
    )
    kwargs.update(overrides)
    return build_tls_config(**kwargs)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

class TestBuildTlsConfigReturnType:

    def test_returns_zone_map(self):
        zm = _default_zone_map()
        assert isinstance(zm, ZoneMap)

    def test_zone1_is_zone_bounds(self):
        zm = _default_zone_map()
        assert isinstance(zm.zone1, ZoneBounds)

    def test_zone2_is_zone_bounds(self):
        zm = _default_zone_map()
        assert isinstance(zm.zone2, ZoneBounds)

    def test_zone3_is_zone_bounds(self):
        zm = _default_zone_map()
        assert isinstance(zm.zone3, ZoneBounds)


# ---------------------------------------------------------------------------
# 2. Metadata fields
# ---------------------------------------------------------------------------

class TestMetadataFields:

    def test_star_id_preserved(self):
        zm = _default_zone_map(star_id="TIC_99999999")
        assert zm.star_id == "TIC_99999999"

    def test_n_samples_preserved(self):
        zm = _default_zone_map(n_samples=5000)
        assert zm.n_samples == 5000

    def test_seed_preserved(self):
        zm = _default_zone_map(seed=123)
        assert zm.seed == 123

    def test_baseline_preserved(self):
        zm = _default_zone_map(observation_baseline_days=365.0)
        assert zm.observation_baseline_days == pytest.approx(365.0)

    def test_min_transits_preserved(self):
        zm = _default_zone_map(min_transits=3)
        assert zm.min_transits == 3

    def test_min_transit_depth_preserved(self):
        zm = _default_zone_map(min_transit_depth=500e-6)
        assert zm.min_transit_depth == pytest.approx(500e-6)

    def test_catalog_source_preserved(self):
        zm = _default_zone_map(catalog_source="Gaia")
        assert zm.catalog_source == "Gaia"

    def test_generated_at_is_recent(self):
        zm = _default_zone_map()
        age = datetime.now(timezone.utc) - zm.generated_at.replace(tzinfo=timezone.utc)
        assert age.total_seconds() < 10

    def test_generated_at_is_datetime(self):
        zm = _default_zone_map()
        assert isinstance(zm.generated_at, datetime)


# ---------------------------------------------------------------------------
# 3. Zone bounds pass through unchanged
# ---------------------------------------------------------------------------

class TestZoneBoundsPassthrough:

    def test_zone1_period_bounds_preserved(self):
        z1, z2, z3 = _make_zones()
        zm = build_tls_config(
            zone1=z1, zone2=z2, zone3=z3,
            star_id="TIC_25155310", n_samples=1000, seed=42,
            observation_baseline_days=27.0, min_transits=2,
            min_transit_depth=200e-6, catalog_source="TIC-8",
        )
        assert zm.zone1.period_min == pytest.approx(z1.period_min)
        assert zm.zone1.period_max == pytest.approx(z1.period_max)

    def test_zone2_duration_bounds_preserved(self):
        z1, z2, z3 = _make_zones()
        zm = build_tls_config(
            zone1=z1, zone2=z2, zone3=z3,
            star_id="TIC_25155310", n_samples=1000, seed=42,
            observation_baseline_days=27.0, min_transits=2,
            min_transit_depth=200e-6, catalog_source="TIC-8",
        )
        assert zm.zone2.duration_min == pytest.approx(z2.duration_min)
        assert zm.zone2.duration_max == pytest.approx(z2.duration_max)

    def test_zone3_period_min_is_none(self):
        _, _, z3 = _make_zones()
        zm = _default_zone_map()
        assert zm.zone3.period_min is None


# ---------------------------------------------------------------------------
# 4. TLS usage pattern
# ---------------------------------------------------------------------------

class TestTlsUsagePattern:

    def test_tls_period_access(self):
        """Simulate how TLS would consume zone1 period bounds."""
        zm = _default_zone_map()
        tls_period_min = zm.zone1.period_min
        tls_period_max = zm.zone1.period_max
        assert tls_period_min is not None
        assert tls_period_max is not None
        assert tls_period_min < tls_period_max

    def test_tls_duration_access(self):
        zm = _default_zone_map()
        assert zm.zone1.duration_min < zm.zone1.duration_max

    def test_tls_depth_access(self):
        zm = _default_zone_map()
        assert zm.zone1.depth_min < zm.zone1.depth_max

    def test_full_pipeline_solar_star(self):
        """End-to-end: sampler → classifier → config for a solar-type star."""
        dist = run_monte_carlo(STELLAR_PARAMS_SOLAR, 27.0, n_samples=10_000, seed=42)
        z1, z2, z3 = classify_zones(dist)
        zm = build_tls_config(
            zone1=z1, zone2=z2, zone3=z3,
            star_id="TIC_25155310",
            n_samples=10_000, seed=42,
            observation_baseline_days=27.0,
            min_transits=2,
            min_transit_depth=200e-6,
            catalog_source="TIC-8",
        )
        assert isinstance(zm, ZoneMap)
        assert zm.zone1.period_min > 0
        assert zm.zone1.period_max <= 13.5
        assert zm.zone1.duration_min < zm.zone1.duration_max
        assert zm.zone1.depth_min < zm.zone1.depth_max < 1.0
