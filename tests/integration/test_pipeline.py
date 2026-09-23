"""
test_pipeline.py — Integration Tests for Full PCE Pipeline

Tests the complete pipeline end-to-end:
    fetcher → sampler → classifier → config → ZoneMap

All catalog queries mocked — no real network I/O.
Uses confirmed planet parameters from mock_data.py to verify
the pipeline produces physically sensible zone maps.
"""

import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, "src")

from astropy import units as u
from pce import run
from pce.schemas import ZoneMap, ZoneBounds
from validation.validate_confirmed_planets import (
    validate_against_confirmed,
    check_planet_in_zones,
    validate_one_star,
)
from tests.fixtures.mock_data import (
    MOCK_TIC8_ROW_SOLAR,
    MOCK_TIC8_ROW_MDWARF,
    STELLAR_PARAMS_SOLAR,
    CONFIRMED_PLANETS,
    make_stellar_params,
)


# ---------------------------------------------------------------------------
# 1. pce.run() end-to-end with mocked fetcher
# ---------------------------------------------------------------------------

class TestPceRunEndToEnd:

    def _mock_fetch(self, row=MOCK_TIC8_ROW_SOLAR):
        return patch("pce.fetcher._query_tic8", return_value=row)

    def test_run_returns_zone_map(self, tmp_path):
        with self._mock_fetch():
            zm = run("TIC_25155310", observation_baseline_days=27.0,
                     n_mc_samples=1000, seed=42)
        assert isinstance(zm, ZoneMap)

    def test_run_star_id_normalised(self, tmp_path):
        with self._mock_fetch():
            zm = run("25155310", observation_baseline_days=27.0,
                     n_mc_samples=1000, seed=42)
        assert zm.star_id == "TIC_25155310"

    def test_run_metadata_correct(self, tmp_path):
        with self._mock_fetch():
            zm = run("TIC_25155310", observation_baseline_days=27.0,
                     min_transits=2, min_transit_depth=200e-6,
                     n_mc_samples=1000, seed=99)
        assert zm.observation_baseline_days == pytest.approx(27.0)
        assert zm.min_transits == 2
        assert zm.min_transit_depth == pytest.approx(200e-6)
        assert zm.n_samples == 1000
        assert zm.seed == 99

    def test_run_reproducible(self, tmp_path):
        """Same seed → same zone bounds."""
        with self._mock_fetch():
            zm1 = run("TIC_25155310", 27.0, n_mc_samples=1000, seed=42,
                      force_refetch=True)
        with self._mock_fetch():
            zm2 = run("TIC_25155310", 27.0, n_mc_samples=1000, seed=42,
                      force_refetch=True)
        assert zm1.zone1.period_min == pytest.approx(zm2.zone1.period_min)
        assert zm1.zone1.period_max == pytest.approx(zm2.zone1.period_max)

    def test_run_different_seeds_differ(self, tmp_path):
        with self._mock_fetch():
            zm1 = run(
                "TIC_25155310",
                27.0,
                n_mc_samples=1000,
                seed=1,
                force_refetch=True,
            )

        with self._mock_fetch():
            zm2 = run(
                "TIC_25155310",
                27.0,
                n_mc_samples=1000,
                seed=2,
                force_refetch=True,
            )

        # Deterministic physical bounds remain identical across seeds.
        assert zm1.zone1.period_min == zm2.zone1.period_min
        assert zm1.zone1.period_max == zm2.zone1.period_max
        assert zm1.zone1.depth_min == zm2.zone1.depth_min

        # Bounds derived from Monte Carlo stellar uncertainty must vary.
        assert zm1.zone1.duration_min != zm2.zone1.duration_min
        assert zm1.zone1.duration_max != zm2.zone1.duration_max
        assert zm1.zone1.depth_max != zm2.zone1.depth_max

    def test_run_zone1_inside_zone2(self, tmp_path):
        with self._mock_fetch():
            zm = run("TIC_25155310", 27.0, n_mc_samples=1000, seed=42)
        assert zm.zone1.period_min >= zm.zone2.period_min
        assert zm.zone1.period_max <= zm.zone2.period_max
        assert zm.zone1.duration_min >= zm.zone2.duration_min
        assert zm.zone1.duration_max <= zm.zone2.duration_max

    def test_run_zone1_period_within_baseline(self, tmp_path):
        """Zone 1 period_max can't exceed observation_baseline / min_transits."""
        with self._mock_fetch():
            zm = run("TIC_25155310", 27.0, min_transits=2,
                     n_mc_samples=1000, seed=42)
        assert zm.zone1.period_max <= 13.5

    def test_run_zone1_physical_bounds(self, tmp_path):
        with self._mock_fetch():
            zm = run("TIC_25155310", 27.0, n_mc_samples=1000, seed=42)
        assert zm.zone1.period_min > 0
        assert zm.zone1.duration_min > 0
        assert zm.zone1.depth_min > 0
        assert zm.zone1.depth_max < 1.0

    def test_run_mdwarf(self, tmp_path):
        with patch("pce.fetcher._query_tic8", return_value=MOCK_TIC8_ROW_MDWARF):
            zm = run("TIC_99999002", 27.0, n_mc_samples=1000, seed=42)
        assert isinstance(zm, ZoneMap)
        assert zm.zone1.period_min > 0
        assert zm.zone1.period_max <= 13.5

    def test_run_three_transits_smaller_pmax(self, tmp_path):
        """Requiring 3 transits should give smaller period_max than 2."""
        with self._mock_fetch():
            zm2 = run("TIC_25155310", 27.0, min_transits=2,
                      n_mc_samples=1000, seed=42, force_refetch=True)
        with self._mock_fetch():
            zm3 = run("TIC_25155310", 27.0, min_transits=3,
                      n_mc_samples=1000, seed=42, force_refetch=True)
        assert zm3.zone1.period_max < zm2.zone1.period_max

    def test_run_longer_baseline_larger_pmax(self, tmp_path):
        with self._mock_fetch():
            zm_short = run("TIC_25155310", 27.0,  n_mc_samples=1000,
                           seed=42, force_refetch=True)
        with self._mock_fetch():
            zm_long  = run("TIC_25155310", 365.0, n_mc_samples=1000,
                           seed=42, force_refetch=True)
        assert zm_long.zone1.period_max > zm_short.zone1.period_max


# ---------------------------------------------------------------------------
# 2. check_planet_in_zones
# ---------------------------------------------------------------------------

class TestCheckPlanetInZones:

    @pytest.fixture
    def zone_map(self):
        """Build a ZoneMap with known zone bounds for deterministic tests."""
        z1 = ZoneBounds(
            period_min=1.0, period_max=10.0,
            duration_min=1.0, duration_max=5.0,
            depth_min=100e-6, depth_max=0.05,
        )
        z2 = ZoneBounds(
            period_min=0.5, period_max=12.0,
            duration_min=0.5, duration_max=7.0,
            depth_min=50e-6, depth_max=0.10,
        )
        z3 = ZoneBounds(
            period_min=None, period_max=0.5,
            duration_min=None, duration_max=0.5,
            depth_min=None, depth_max=50e-6,
        )
        return ZoneMap(
            star_id="TIC_TEST",
            zone1=z1, zone2=z2, zone3=z3,
            n_samples=1000, seed=42,
            observation_baseline_days=27.0,
            min_transits=2, min_transit_depth=100e-6,
            catalog_source="TIC-8",
        )

    def test_planet_inside_zone1(self, zone_map):
        r = check_planet_in_zones(5.0, 3.0, 0.01, zone_map)
        assert r["overall"] == "zone1"
        assert r["period_zone"] == 1
        assert r["duration_zone"] == 1
        assert r["depth_zone"] == 1

    def test_planet_inside_zone2_only(self, zone_map):
        """Period in Zone 2 but outside Zone 1."""
        r = check_planet_in_zones(11.0, 3.0, 0.01, zone_map)
        assert r["overall"] == "zone2"
        assert r["period_zone"] == 2

    def test_planet_miss_period(self, zone_map):
        """Period outside Zone 2."""
        r = check_planet_in_zones(13.0, 3.0, 0.01, zone_map)
        assert r["overall"] == "miss"
        assert r["period_zone"] == 0

    def test_planet_miss_depth(self, zone_map):
        """Depth above Zone 2 max."""
        r = check_planet_in_zones(5.0, 3.0, 0.20, zone_map)
        assert r["overall"] == "miss"
        assert r["depth_zone"] == 0

    def test_planet_on_zone1_boundary(self, zone_map):
        """Exactly on Zone 1 boundary counts as Zone 1."""
        r = check_planet_in_zones(1.0, 1.0, 100e-6, zone_map)
        assert r["period_zone"] == 1
        assert r["duration_zone"] == 1
        assert r["depth_zone"] == 1


# ---------------------------------------------------------------------------
# 3. validate_against_confirmed with mocked pipeline
# ---------------------------------------------------------------------------

class TestValidateAgainstConfirmed:

    def _make_zone_map_for(self, planet: dict) -> ZoneMap:
        """
        Build a ZoneMap that correctly contains the given planet's parameters.
        Used to mock pce.run() so validate_one_star sees a hit.
        """
        p = planet["period_day"]
        d = planet["duration_hr"]
        depth = planet["depth"]

        z1 = ZoneBounds(
            period_min=p * 0.5,  period_max=p * 2.0,
            duration_min=d * 0.3, duration_max=d * 3.0,
            depth_min=depth * 0.1, depth_max=min(depth * 10.0, 0.99),
        )
        z2 = ZoneBounds(
            period_min=p * 0.2,  period_max=p * 3.0,
            duration_min=d * 0.1, duration_max=d * 5.0,
            depth_min=depth * 0.05, depth_max=min(depth * 20.0, 0.99),
        )
        z3 = ZoneBounds(period_min=None, period_max=p * 0.2)
        return ZoneMap(
            star_id=planet["star_id"],
            zone1=z1, zone2=z2, zone3=z3,
            n_samples=1000, seed=42,
            observation_baseline_days=27.0,
            min_transits=2, min_transit_depth=200e-6,
            catalog_source="TIC-8",
        )

    def test_all_hits_gives_100_percent(self):
        """When every planet lands in Zone 1, success rate = 1.0."""
        planets = CONFIRMED_PLANETS[:3]

        def mock_run(star_id, **kwargs):
            planet = next(p for p in planets if p["star_id"] == star_id)
            return self._make_zone_map_for(planet)

        with patch("validation.validate_confirmed_planets.run", side_effect=mock_run):
            summary = validate_against_confirmed(planets=planets, verbose=False)

        assert summary["success_rate"] == pytest.approx(1.0)
        assert summary["n_miss"] == 0
        assert summary["n_error"] == 0

    def test_all_misses_gives_zero_percent(self):
        """When every planet lands outside Zone 2, success rate = 0.0."""
        planets = CONFIRMED_PLANETS[:3]

        def mock_run(star_id, **kwargs):
            planet = next(p for p in planets if p["star_id"] == star_id)
            p = planet["period_day"]
            z1 = ZoneBounds(period_min=p*10, period_max=p*20,
                            duration_min=100.0, duration_max=200.0,
                            depth_min=0.5, depth_max=0.9)
            z2 = ZoneBounds(period_min=p*8, period_max=p*25,
                            duration_min=80.0, duration_max=300.0,
                            depth_min=0.4, depth_max=0.95)
            z3 = ZoneBounds(period_min=None, period_max=p*8)
            return ZoneMap(
                star_id=star_id, zone1=z1, zone2=z2, zone3=z3,
                n_samples=1000, seed=42, observation_baseline_days=27.0,
                min_transits=2, min_transit_depth=200e-6, catalog_source="TIC-8",
            )

        with patch("validation.validate_confirmed_planets.run", side_effect=mock_run):
            summary = validate_against_confirmed(planets=planets, verbose=False)

        assert summary["success_rate"] == pytest.approx(0.0)
        assert summary["n_miss"] == len(planets)

    def test_error_star_counted_separately(self):
        """Stars that raise exceptions are counted in n_error, not n_miss."""
        planets = CONFIRMED_PLANETS[:2]

        def mock_run(star_id, **kwargs):
            raise RuntimeError("catalog unavailable")

        with patch("validation.validate_confirmed_planets.run", side_effect=mock_run):
            summary = validate_against_confirmed(planets=planets, verbose=False)

        assert summary["n_error"] == len(planets)
        assert summary["n_miss"] == 0

    def test_summary_keys_present(self):
        planets = CONFIRMED_PLANETS[:1]

        def mock_run(star_id, **kwargs):
            return self._make_zone_map_for(planets[0])

        with patch("validation.validate_confirmed_planets.run", side_effect=mock_run):
            summary = validate_against_confirmed(planets=planets, verbose=False)

        for key in ["n_total", "n_zone1", "n_zone2", "n_miss", "n_error",
                    "success_rate", "results"]:
            assert key in summary

    def test_n_total_matches_input(self):
        planets = CONFIRMED_PLANETS[:3]

        def mock_run(star_id, **kwargs):
            planet = next(p for p in planets if p["star_id"] == star_id)
            return self._make_zone_map_for(planet)

        with patch("validation.validate_confirmed_planets.run", side_effect=mock_run):
            summary = validate_against_confirmed(planets=planets, verbose=False)

        assert summary["n_total"] == 3

    def test_csv_written_when_path_given(self, tmp_path):
        planets = CONFIRMED_PLANETS[:2]
        csv_path = tmp_path / "results.csv"

        def mock_run(star_id, **kwargs):
            planet = next(p for p in planets if p["star_id"] == star_id)
            return self._make_zone_map_for(planet)

        with patch("validation.validate_confirmed_planets.run", side_effect=mock_run):
            validate_against_confirmed(
                planets=planets, verbose=False, output_csv=csv_path
            )

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "star_id" in content
        assert "overall" in content
