"""
test_physics.py — Unit Tests for physics.py

Coverage:
    - roche_limit_period
    - semi_major_axis
    - transit_duration_central
    - transit_duration_grazing_onset
    - transit_depth
    - minimum_detectable_planet_radius
    - maximum_period_from_baseline

Each function gets:
    - Known-value accuracy test(s)
    - Edge case(s)
    - Invalid input rejection test(s)
"""

import math
import numpy as np
import pytest
from astropy import units as u
from astropy.units import Quantity

import sys
sys.path.insert(0, "src")

from pce.physics import (
    roche_limit_period,
    semi_major_axis,
    transit_duration_central,
    transit_duration_grazing_onset,
    transit_depth,
    minimum_detectable_planet_radius,
    maximum_period_from_baseline,
    _vectorized_minimum_detectable_planet_radius,
    _vectorized_transit_depth,
    _vectorized_transit_duration_central,
    _vectorized_transit_duration_grazing_onset,
    _vectorized_semi_major_axis,
    R_PLANET_MAX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_quantity(q: Quantity, expected: Quantity, rtol: float = 1e-3) -> bool:
    """Return True if q is within rtol of expected (same physical type)."""
    ratio = (q.to(expected.unit).value / expected.value) - 1.0
    return abs(ratio) < rtol


# ---------------------------------------------------------------------------
# 1. roche_limit_period
# ---------------------------------------------------------------------------

class TestRocheLimitPeriod:

    def test_saturn_density_range(self):
        """Saturn density (~687 kg/m^3) should give Roche period ~14-16 hr."""
        rho = 687 * u.kg / u.m**3
        result = roche_limit_period(rho)
        assert result.to(u.hour).value == pytest.approx(15.2, rel=0.01)

    def test_reference_density_gives_prefactor(self):
        """At rho = 1 g/cm^3 the formula reduces to exactly 12.6 hr."""
        rho = 1.0 * u.g / u.cm**3
        result = roche_limit_period(rho)
        assert result.to(u.hour).value == pytest.approx(12.6, rel=1e-6)

    def test_higher_density_shorter_period(self):
        """Denser planet should have shorter Roche period."""
        rho_low  = 500  * u.kg / u.m**3
        rho_high = 5000 * u.kg / u.m**3
        assert roche_limit_period(rho_high) < roche_limit_period(rho_low)

    def test_iron_planet_density(self):
        """Iron-rich planet (~10 000 kg/m^3 = 10 g/cm^3) — period should be < 4 hr."""
        rho = 10_000 * u.kg / u.m**3
        result = roche_limit_period(rho)
        assert result.to(u.hour).value < 4.0

    def test_returns_quantity_in_hours(self):
        """Output must be an astropy Quantity convertible to hours."""
        rho = 1.0 * u.g / u.cm**3
        result = roche_limit_period(rho)
        assert isinstance(result, Quantity)
        _ = result.to(u.hour)  # must not raise

    def test_unit_invariance(self):
        """Same density in different units should give the same result."""
        rho_si  = 1000.0 * u.kg / u.m**3
        rho_cgs = rho_si.to(u.g / u.cm**3)
        p_si  = roche_limit_period(rho_si).to(u.hour).value
        p_cgs = roche_limit_period(rho_cgs).to(u.hour).value
        assert p_si == pytest.approx(p_cgs, rel=1e-9)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            roche_limit_period(0.0 * u.g / u.cm**3)

    def test_negative_density_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            roche_limit_period(-1.0 * u.g / u.cm**3)


# ---------------------------------------------------------------------------
# 2. semi_major_axis
# ---------------------------------------------------------------------------

class TestSemiMajorAxis:

    def test_earth_period_gives_1_au(self):
        """Earth's period (365.25 d) around solar mass → 1 AU."""
        a = semi_major_axis(365.25 * u.day, 1.0 * u.M_sun)
        assert a.to(u.au).value == pytest.approx(1.0, rel=1e-3)

    def test_shorter_period_smaller_orbit(self):
        """Shorter period must correspond to smaller semi-major axis."""
        M = 1.0 * u.M_sun
        a_short = semi_major_axis(1.0  * u.day, M)
        a_long  = semi_major_axis(10.0 * u.day, M)
        assert a_short < a_long

    def test_kepler_third_law_scaling(self):
        """a^3 proportional to P^2 * M — test with known ratio."""
        M  = 1.0 * u.M_sun
        P1 = 1.0 * u.day
        P2 = 8.0 * u.day
        a1 = semi_major_axis(P1, M)
        a2 = semi_major_axis(P2, M)
        # a2/a1 = (P2/P1)^(2/3) = 8^(2/3) = 4
        ratio = a2.to(u.m).value / a1.to(u.m).value
        assert ratio == pytest.approx(4.0, rel=1e-6)

    def test_heavier_star_larger_orbit(self):
        """More massive star → larger semi-major axis for same period."""
        P = 10.0 * u.day
        a_low  = semi_major_axis(P, 0.5 * u.M_sun)
        a_high = semi_major_axis(P, 2.0 * u.M_sun)
        assert a_high > a_low

    def test_returns_quantity(self):
        result = semi_major_axis(365.25 * u.day, 1.0 * u.M_sun)
        assert isinstance(result, Quantity)
        _ = result.to(u.au)

    def test_zero_period_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            semi_major_axis(0.0 * u.day, 1.0 * u.M_sun)

    def test_negative_period_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            semi_major_axis(-5.0 * u.day, 1.0 * u.M_sun)

    def test_zero_mass_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            semi_major_axis(10.0 * u.day, 0.0 * u.M_sun)

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            semi_major_axis(10.0 * u.day, -1.0 * u.M_sun)


# ---------------------------------------------------------------------------
# 3. transit_duration_central
# ---------------------------------------------------------------------------

class TestTransitDurationCentral:

    def _solar_earth_duration(self):
        return transit_duration_central(
            period=365.25 * u.day,
            star_radius=1.0 * u.R_sun,
            planet_radius=1.0 * u.R_earth,
            star_mass=1.0 * u.M_sun,
        )

    def test_earth_transit_duration_approx_13hr(self):
        """Earth transiting the Sun → ~13 hr."""
        dur = self._solar_earth_duration()
        assert dur.to(u.hour).value == pytest.approx(13.1, rel=0.01)

    def test_central_longer_than_grazing(self):
        """Central transit must be longer than grazing onset for same params."""
        kwargs = dict(
            period=10.0 * u.day,
            star_radius=1.0 * u.R_sun,
            planet_radius=1.0 * u.R_jup,
            star_mass=1.0 * u.M_sun,
        )
        dur_c = transit_duration_central(**kwargs)
        dur_g = transit_duration_grazing_onset(**kwargs)
        assert dur_c > dur_g

    def test_longer_period_longer_duration(self):
        """Larger orbit → slower crossing → longer transit."""
        common = dict(
            star_radius=1.0 * u.R_sun,
            planet_radius=1.0 * u.R_earth,
            star_mass=1.0 * u.M_sun,
        )
        d1 = transit_duration_central(period=5.0  * u.day, **common)
        d2 = transit_duration_central(period=50.0 * u.day, **common)
        assert d2 > d1

    def test_larger_planet_longer_duration(self):
        """Larger planet → wider chord → longer transit."""
        common = dict(
            period=10.0 * u.day,
            star_radius=1.0 * u.R_sun,
            star_mass=1.0 * u.M_sun,
        )
        d_small = transit_duration_central(planet_radius=1.0 * u.R_earth, **common)
        d_large = transit_duration_central(planet_radius=1.0 * u.R_jup,   **common)
        assert d_large > d_small

    def test_returns_quantity_in_hours(self):
        dur = self._solar_earth_duration()
        assert isinstance(dur, Quantity)
        _ = dur.to(u.hour)

    def test_zero_star_radius_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            transit_duration_central(10*u.day, 0*u.R_sun, 1*u.R_earth, 1*u.M_sun)

    def test_zero_planet_radius_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            transit_duration_central(10*u.day, 1*u.R_sun, 0*u.R_earth, 1*u.M_sun)

    def test_planet_inside_star_raises(self):
        """Very short period — planet orbit inside star — should raise."""
        with pytest.raises(ValueError, match="inside the star"):
            transit_duration_central(
                period=0.001 * u.day,
                star_radius=10.0 * u.R_sun,
                planet_radius=1.0 * u.R_jup,
                star_mass=1.0 * u.M_sun,
            )


# ---------------------------------------------------------------------------
# 4. transit_duration_grazing_onset
# ---------------------------------------------------------------------------

class TestTransitDurationGrazingOnset:

    def test_grazing_shorter_than_central(self):
        """Grazing onset duration must be < central for same params."""
        kwargs = dict(
            period=10.0 * u.day,
            star_radius=1.0 * u.R_sun,
            planet_radius=1.0 * u.R_jup,
            star_mass=1.0 * u.M_sun,
        )
        assert transit_duration_grazing_onset(**kwargs) < transit_duration_central(**kwargs)

    def test_earth_grazing_positive(self):
        """Grazing duration for Earth transiting Sun must be > 0."""
        dur = transit_duration_grazing_onset(
            365.25 * u.day, 1.0 * u.R_sun, 1.0 * u.R_earth, 1.0 * u.M_sun
        )
        assert dur.to(u.hour).value > 0

    def test_larger_planet_longer_grazing(self):
        """Larger planet → larger grazing chord → longer grazing duration."""
        common = dict(period=10*u.day, star_radius=1*u.R_sun, star_mass=1*u.M_sun)
        d_small = transit_duration_grazing_onset(planet_radius=1*u.R_earth, **common)
        d_large = transit_duration_grazing_onset(planet_radius=1*u.R_jup,   **common)
        assert d_large > d_small

    def test_returns_quantity_in_hours(self):
        dur = transit_duration_grazing_onset(
            10*u.day, 1*u.R_sun, 1*u.R_earth, 1*u.M_sun
        )
        assert isinstance(dur, Quantity)
        _ = dur.to(u.hour)

    def test_zero_star_radius_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            transit_duration_grazing_onset(10*u.day, 0*u.R_sun, 1*u.R_earth, 1*u.M_sun)

    def test_zero_planet_radius_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            transit_duration_grazing_onset(10*u.day, 1*u.R_sun, 0*u.R_earth, 1*u.M_sun)


# ---------------------------------------------------------------------------
# 5. transit_depth
# ---------------------------------------------------------------------------

class TestTransitDepth:

    def test_earth_depth_approx_84_ppm(self):
        """Earth transiting the Sun → ~84 ppm."""
        depth = transit_depth(1.0 * u.R_earth, 1.0 * u.R_sun)
        assert depth * 1e6 == pytest.approx(84.0, rel=0.01)

    def test_jupiter_depth_approx_1_percent(self):
        """Jupiter transiting the Sun → ~1.1%."""
        depth = transit_depth(1.0 * u.R_jup, 1.0 * u.R_sun)
        assert depth * 100 == pytest.approx(1.06, rel=0.02)

    def test_depth_scales_as_radius_squared(self):
        """Double the planet radius → 4× the depth."""
        R_s = 1.0 * u.R_sun
        d1 = transit_depth(1.0 * u.R_earth, R_s)
        d2 = transit_depth(2.0 * u.R_earth, R_s)
        assert d2 / d1 == pytest.approx(4.0, rel=1e-9)

    def test_depth_between_zero_and_one(self):
        depth = transit_depth(1.0 * u.R_jup, 1.0 * u.R_sun)
        assert 0 < depth < 1

    def test_returns_float(self):
        depth = transit_depth(1.0 * u.R_earth, 1.0 * u.R_sun)
        assert isinstance(depth, float)

    def test_unit_invariance(self):
        """Same radii in metres vs solar/earth units → same depth."""
        R_s_sun = 1.0 * u.R_sun
        R_p_earth = 1.0 * u.R_earth
        d1 = transit_depth(R_p_earth, R_s_sun)
        d2 = transit_depth(R_p_earth.to(u.m), R_s_sun.to(u.m))
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_zero_planet_radius_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            transit_depth(0.0 * u.R_earth, 1.0 * u.R_sun)

    def test_zero_star_radius_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            transit_depth(1.0 * u.R_earth, 0.0 * u.R_sun)

    def test_planet_larger_than_star_raises(self):
        with pytest.raises(ValueError, match="depth would exceed 1.0"):
            transit_depth(2.0 * u.R_sun, 1.0 * u.R_sun)


# ---------------------------------------------------------------------------
# 6. minimum_detectable_planet_radius
# ---------------------------------------------------------------------------

class TestMinimumDetectablePlanetRadius:

    def test_200ppm_solar_star(self):
        """200 ppm floor around solar star → ~1.54 R_earth."""
        R_p_min = minimum_detectable_planet_radius(1.0 * u.R_sun, 200e-6)
        assert R_p_min.to(u.R_earth).value == pytest.approx(1.543, rel=0.01)

    def test_larger_star_larger_minimum(self):
        """Larger star → R_p_min scales with R_star."""
        floor = 200e-6
        r1 = minimum_detectable_planet_radius(1.0 * u.R_sun, floor)
        r2 = minimum_detectable_planet_radius(2.0 * u.R_sun, floor)
        assert r2.to(u.m).value == pytest.approx(2 * r1.to(u.m).value, rel=1e-9)

    def test_deeper_floor_larger_minimum(self):
        """Shallower sensitivity (higher floor) → larger minimum radius."""
        R_s = 1.0 * u.R_sun
        r_sensitive = minimum_detectable_planet_radius(R_s, 100e-6)
        r_insensitive = minimum_detectable_planet_radius(R_s, 500e-6)
        assert r_insensitive > r_sensitive

    def test_returns_quantity(self):
        result = minimum_detectable_planet_radius(1.0 * u.R_sun, 200e-6)
        assert isinstance(result, Quantity)

    def test_output_unit_matches_input_unit(self):
        """Output should carry same unit as star_radius input."""
        result_sun   = minimum_detectable_planet_radius(1.0 * u.R_sun, 200e-6)
        result_metre = minimum_detectable_planet_radius((1.0 * u.R_sun).to(u.m), 200e-6)
        assert result_sun.to(u.m).value == pytest.approx(result_metre.to(u.m).value, rel=1e-9)

    def test_zero_star_radius_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            minimum_detectable_planet_radius(0.0 * u.R_sun, 200e-6)

    def test_zero_depth_floor_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            minimum_detectable_planet_radius(1.0 * u.R_sun, 0.0)

    def test_negative_depth_floor_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            minimum_detectable_planet_radius(1.0 * u.R_sun, -100e-6)


# ---------------------------------------------------------------------------
# 7. maximum_period_from_baseline
# ---------------------------------------------------------------------------

class TestMaximumPeriodFromBaseline:

    def test_27_day_tess_sector_2_transits(self):
        """27-day TESS sector, 2 transits → 13.5 days."""
        P_max = maximum_period_from_baseline(27.0 * u.day, n_min_transits=2)
        assert P_max.to(u.day).value == pytest.approx(13.5, rel=1e-9)

    def test_27_day_tess_sector_3_transits(self):
        """27-day TESS sector, 3 transits → 9.0 days."""
        P_max = maximum_period_from_baseline(27.0 * u.day, n_min_transits=3)
        assert P_max.to(u.day).value == pytest.approx(9.0, rel=1e-9)

    def test_longer_baseline_larger_pmax(self):
        """Longer baseline → larger P_max."""
        p1 = maximum_period_from_baseline(27.0  * u.day, 2)
        p2 = maximum_period_from_baseline(365.0 * u.day, 2)
        assert p2 > p1

    def test_more_transits_smaller_pmax(self):
        """More required transits → smaller P_max."""
        baseline = 100.0 * u.day
        p2 = maximum_period_from_baseline(baseline, 2)
        p3 = maximum_period_from_baseline(baseline, 3)
        assert p3 < p2

    def test_returns_quantity_in_days(self):
        result = maximum_period_from_baseline(27.0 * u.day, 2)
        assert isinstance(result, Quantity)
        _ = result.to(u.day)

    def test_unit_invariance(self):
        """Baseline in hours vs days → same P_max."""
        p_day  = maximum_period_from_baseline(27.0 * u.day, 2).to(u.day).value
        p_hour = maximum_period_from_baseline(27.0 * 24 * u.hour, 2).to(u.day).value
        assert p_day == pytest.approx(p_hour, rel=1e-9)

    def test_zero_baseline_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            maximum_period_from_baseline(0.0 * u.day, 2)

    def test_negative_baseline_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            maximum_period_from_baseline(-10.0 * u.day, 2)

    def test_one_transit_raises(self):
        with pytest.raises(ValueError, match="must be >= 2"):
            maximum_period_from_baseline(27.0 * u.day, n_min_transits=1)

    def test_zero_transits_raises(self):
        with pytest.raises(ValueError, match="must be >= 2"):
            maximum_period_from_baseline(27.0 * u.day, n_min_transits=0)


# ---------------------------------------------------------------------------
# 8. R_PLANET_MAX constant
# ---------------------------------------------------------------------------

class TestRPlanetMax:

    def test_value_is_2_2_rjup(self):
        assert R_PLANET_MAX.to(u.R_jup).value == pytest.approx(2.2, rel=1e-9)

    def test_is_quantity(self):
        assert isinstance(R_PLANET_MAX, Quantity)


# ---------------------------------------------------------------------------
# Vectorised physics helpers
# ---------------------------------------------------------------------------

def test_vectorized_semi_major_axis_matches_scalar():
    periods_s = np.array([10.0, 20.0, 40.0]) * 86400.0
    masses_kg = np.array([1.0, 0.8, 1.2]) * u.M_sun.to(u.kg)

    result = _vectorized_semi_major_axis(periods_s, masses_kg)
    expected = np.array([
        semi_major_axis(p * u.day, m * u.M_sun).to_value(u.m)
        for p, m in zip([10.0, 20.0, 40.0], [1.0, 0.8, 1.2])
    ])

    assert np.allclose(result, expected)


def test_vectorized_duration_helpers_match_scalar():
    periods_s = np.array([5.0, 10.0, 20.0]) * 86400.0
    radii_m = np.array([1.0, 0.9, 1.1]) * u.R_sun.to(u.m)
    planet_radii_m = np.array([1.0, 1.5, 2.0]) * u.R_earth.to(u.m)
    masses_kg = np.array([1.0, 0.9, 1.1]) * u.M_sun.to(u.kg)

    central = _vectorized_transit_duration_central(
        periods_s, radii_m, planet_radii_m, masses_kg
    )
    grazing = _vectorized_transit_duration_grazing_onset(
        periods_s, radii_m, planet_radii_m, masses_kg
    )

    expected_central = np.array([
        transit_duration_central(
            p * u.day, r * u.R_sun, rp * u.R_earth, m * u.M_sun
        ).to_value(u.hour)
        for p, r, rp, m in zip(
            [5.0, 10.0, 20.0], [1.0, 0.9, 1.1], [1.0, 1.5, 2.0], [1.0, 0.9, 1.1]
        )
    ])
    expected_grazing = np.array([
        transit_duration_grazing_onset(
            p * u.day, r * u.R_sun, rp * u.R_earth, m * u.M_sun
        ).to_value(u.hour)
        for p, r, rp, m in zip(
            [5.0, 10.0, 20.0], [1.0, 0.9, 1.1], [1.0, 1.5, 2.0], [1.0, 0.9, 1.1]
        )
    ])

    assert np.allclose(central, expected_central)
    assert np.allclose(grazing, expected_grazing)
    assert np.all(grazing < central)


def test_vectorized_depth_and_minimum_radius_match_scalar():
    star_radii_m = np.array([0.8, 1.0, 1.2]) * u.R_sun.to(u.m)
    planet_radii_m = np.array([1.0, 2.0, 3.0]) * u.R_earth.to(u.m)
    depth_floor = 200e-6

    depths = _vectorized_transit_depth(planet_radii_m, star_radii_m)
    min_radii = _vectorized_minimum_detectable_planet_radius(
        star_radii_m, depth_floor
    )

    expected_depths = np.array([
        transit_depth(rp * u.R_earth, rs * u.R_sun)
        for rp, rs in zip([1.0, 2.0, 3.0], [0.8, 1.0, 1.2])
    ])
    expected_min_radii = np.array([
        minimum_detectable_planet_radius(rs * u.R_sun, depth_floor).to_value(u.m)
        for rs in [0.8, 1.0, 1.2]
    ])

    assert np.allclose(depths, expected_depths)
    assert np.allclose(min_radii, expected_min_radii)


def test_vectorized_duration_returns_nan_for_invalid_geometry():
    periods_s = np.array([1.0]) * 86400.0
    radii_m = np.array([10.0]) * u.R_sun.to(u.m)
    planet_radii_m = np.array([1.0]) * u.R_earth.to(u.m)
    masses_kg = np.array([1.0]) * u.M_sun.to(u.kg)

    result = _vectorized_transit_duration_central(
        periods_s, radii_m, planet_radii_m, masses_kg
    )

    assert np.isnan(result[0])
