"""
test_fetcher.py — Unit Tests for fetcher.py

All TIC-8 and Gaia network calls are mocked — no real network I/O.
Tests cover:
    - _normalise_tic_id
    - _extract_tic8 (TIC-8 happy path, asymmetric errors, bad data)
    - _extract_gaia (Gaia happy path, percentile → sigma conversion)
    - Cache: save, load, invalidate, stale, force_refetch
    - fetch_stellar_params (end-to-end with mocked catalog calls)
    - Validation rejection (negative mass, missing uncertainties)
"""

import math
import sys
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")

from astropy import units as u
from pce.schemas import StellarParameters
from pce.fetcher import (
    fetch_stellar_params,
    _normalise_tic_id,
    _extract_tic8,
    _extract_gaia,
    _query_tic8,
)
from utils.cache import save_to_cache, load_from_cache, invalidate_cache
from tests.fixtures.mock_data import (
    MOCK_TIC8_ROW_SOLAR,
    MOCK_TIC8_ROW_ASYMMETRIC,
    MOCK_TIC8_ROW_MDWARF,
    MOCK_TIC8_ROW_BAD_NEGATIVE_MASS,
    MOCK_TIC8_ROW_MISSING_UNCERTAINTIES,
    MOCK_GAIA_ROW_SOLAR,
    make_stellar_params,
)


# ---------------------------------------------------------------------------
# Fixture: temp cache directory (isolated per test)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache(tmp_path):
    return tmp_path / "cache"


# ---------------------------------------------------------------------------
# 1. _normalise_tic_id
# ---------------------------------------------------------------------------

class TestNormaliseTicId:

    def test_plain_number(self):
        assert _normalise_tic_id("25155310") == "TIC_25155310"

    def test_tic_prefix(self):
        assert _normalise_tic_id("TIC25155310") == "TIC_25155310"

    def test_tic_underscore_prefix(self):
        assert _normalise_tic_id("TIC_25155310") == "TIC_25155310"

    def test_lowercase_accepted(self):
        assert _normalise_tic_id("tic_25155310") == "TIC_25155310"

    def test_whitespace_stripped(self):
        assert _normalise_tic_id("  25155310  ") == "TIC_25155310"

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _normalise_tic_id("HD_209458")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _normalise_tic_id("")


# ---------------------------------------------------------------------------
# 2. _extract_tic8
# ---------------------------------------------------------------------------

class TestExtractTic8:

    def test_solar_extraction_values(self):
        """Values extracted from solar mock match expected quantities."""
        params = _extract_tic8("TIC_25155310", MOCK_TIC8_ROW_SOLAR)
        assert params["mass"].to(u.M_sun).value == pytest.approx(1.03, rel=1e-9)
        assert params["radius"].to(u.R_sun).value == pytest.approx(1.13, rel=1e-9)
        assert params["teff"].to(u.K).value == pytest.approx(5765.0, rel=1e-9)
        assert params["luminosity"].to(u.L_sun).value == pytest.approx(1.35, rel=1e-9)

    def test_solar_extraction_uncertainties(self):
        """Uncertainties extracted correctly as positive Quantities."""
        params = _extract_tic8("TIC_25155310", MOCK_TIC8_ROW_SOLAR)
        assert params["mass_err_lo"].to(u.M_sun).value == pytest.approx(0.07)
        assert params["mass_err_hi"].to(u.M_sun).value == pytest.approx(0.07)
        assert params["radius_err_lo"].to(u.R_sun).value == pytest.approx(0.05)
        assert params["radius_err_hi"].to(u.R_sun).value == pytest.approx(0.05)

    def test_asymmetric_errors_preserved(self):
        """Asymmetric lo/hi uncertainties extracted independently."""
        params = _extract_tic8("TIC_99999001", MOCK_TIC8_ROW_ASYMMETRIC)
        assert params["mass_err_lo"].to(u.M_sun).value == pytest.approx(0.08)
        assert params["mass_err_hi"].to(u.M_sun).value == pytest.approx(0.15)
        assert params["radius_err_lo"].to(u.R_sun).value == pytest.approx(0.10)
        assert params["radius_err_hi"].to(u.R_sun).value == pytest.approx(0.20)

    def test_mdwarf_extraction(self):
        """M-dwarf values (small mass/radius/teff) extracted correctly."""
        params = _extract_tic8("TIC_99999002", MOCK_TIC8_ROW_MDWARF)
        assert params["mass"].to(u.M_sun).value == pytest.approx(0.35)
        assert params["teff"].to(u.K).value == pytest.approx(3400.0)

    def test_catalog_source_is_tic8(self):
        params = _extract_tic8("TIC_25155310", MOCK_TIC8_ROW_SOLAR)
        assert params["catalog_source"] == "TIC-8"

    def test_tic_id_preserved(self):
        params = _extract_tic8("TIC_25155310", MOCK_TIC8_ROW_SOLAR)
        assert params["tic_id"] == "TIC_25155310"

    def test_all_quantities_have_correct_units(self):
        params = _extract_tic8("TIC_25155310", MOCK_TIC8_ROW_SOLAR)
        params["mass"].to(u.M_sun)
        params["radius"].to(u.R_sun)
        params["teff"].to(u.K)
        params["luminosity"].to(u.L_sun)

    def test_missing_mass_raises(self):
        row = dict(MOCK_TIC8_ROW_SOLAR)
        row["mass"] = None
        with pytest.raises(ValueError, match="mass"):
            _extract_tic8("TIC_25155310", row)

    def test_nan_mass_raises(self):
        row = dict(MOCK_TIC8_ROW_SOLAR)
        row["mass"] = float("nan")
        with pytest.raises(ValueError, match="mass"):
            _extract_tic8("TIC_25155310", row)
    
    def test_masked_mass_raises(self):
        """Astropy/NumPy masked catalog values are rejected cleanly."""
        row = dict(MOCK_TIC8_ROW_SOLAR)
        row["mass"] = np.ma.masked

        with pytest.raises(ValueError, match="mass"):
            _extract_tic8("TIC_25155310", row)

    def test_missing_uncertainty_raises(self):
        row = dict(MOCK_TIC8_ROW_SOLAR)
        row["eneg_Mass"] = None
        with pytest.raises(ValueError, match="TIC-8 uncertainty for 'mass'"):
            _extract_tic8("TIC_25155310", row)

    def test_missing_luminosity_raises(self):
        row = dict(MOCK_TIC8_ROW_SOLAR)
        row["lum"] = float("nan")
        with pytest.raises(ValueError, match="luminosity"):
            _extract_tic8("TIC_25155310", row)


    def test_tic_symmetric_uncertainty_fallback(self):
        """Symmetric e_* uncertainties are used when asymmetric values are unavailable."""
        row = dict(MOCK_TIC8_ROW_SOLAR)

        # Simulate the L 98-59 situation:
        # asymmetric uncertainties are unavailable,
        # but symmetric TIC uncertainties are available.
        row["eneg_Mass"] = float("nan")
        row["epos_Mass"] = float("nan")
        row["e_mass"] = 0.03

        row["eneg_Rad"] = float("nan")
        row["epos_Rad"] = float("nan")
        row["e_rad"] = 0.02

        row["eneg_Teff"] = float("nan")
        row["epos_Teff"] = float("nan")
        row["e_Teff"] = 50.0

        row["eneg_Lum"] = float("nan")
        row["epos_Lum"] = float("nan")
        row["e_lum"] = 0.05

        result = _extract_tic8("TIC_25155310", row)

        assert result["mass_err_lo"].to(u.M_sun).value == pytest.approx(0.03)
        assert result["mass_err_hi"].to(u.M_sun).value == pytest.approx(0.03)

        assert result["radius_err_lo"].to(u.R_sun).value == pytest.approx(0.02)
        assert result["radius_err_hi"].to(u.R_sun).value == pytest.approx(0.02)

        assert result["teff_err_lo"].to(u.K).value == pytest.approx(50.0)
        assert result["teff_err_hi"].to(u.K).value == pytest.approx(50.0)

        assert result["luminosity_err_lo"].to(u.L_sun).value == pytest.approx(0.05)
        assert result["luminosity_err_hi"].to(u.L_sun).value == pytest.approx(0.05)

# ---------------------------------------------------------------------------
# 3. _extract_gaia
# ---------------------------------------------------------------------------

class TestExtractGaia:

    def test_gaia_central_values(self):
        """Central values extracted correctly from Gaia row."""
        params = _extract_gaia("TIC_25155310", MOCK_GAIA_ROW_SOLAR, partial_tic8={})
        assert params["mass"].to(u.M_sun).value == pytest.approx(1.01, rel=1e-9)
        assert params["radius"].to(u.R_sun).value == pytest.approx(1.10, rel=1e-9)
        assert params["teff"].to(u.K).value == pytest.approx(5750.0, rel=1e-9)
        assert params["luminosity"].to(u.L_sun).value == pytest.approx(1.28, rel=1e-9)

    def test_gaia_percentile_to_sigma_conversion(self):
        """
        Gaia 16th/84th percentiles convert correctly to asymmetric uncertainties.
        mass_flame=1.01, lower=0.92, upper=1.12
        err_lo = 1.01 - 0.92 = 0.09
        err_hi = 1.12 - 1.01 = 0.11
        """
        params = _extract_gaia("TIC_25155310", MOCK_GAIA_ROW_SOLAR, partial_tic8={})
        assert params["mass_err_lo"].to(u.M_sun).value == pytest.approx(0.09, rel=1e-6)
        assert params["mass_err_hi"].to(u.M_sun).value == pytest.approx(0.11, rel=1e-6)

    def test_gaia_teff_uncertainties(self):
        """
        teff_gspphot=5750, lower=5680, upper=5820
        err_lo = 5750 - 5680 = 70
        err_hi = 5820 - 5750 = 70
        """
        params = _extract_gaia("TIC_25155310", MOCK_GAIA_ROW_SOLAR, partial_tic8={})
        assert params["teff_err_lo"].to(u.K).value == pytest.approx(70.0, rel=1e-6)
        assert params["teff_err_hi"].to(u.K).value == pytest.approx(70.0, rel=1e-6)

    def test_catalog_source_is_gaia(self):
        params = _extract_gaia("TIC_25155310", MOCK_GAIA_ROW_SOLAR, partial_tic8={})
        assert params["catalog_source"] == "TIC-8+Gaia"
        assert params["catalog_version"] == "TIC-8 + DR3"

    def test_missing_gaia_field_raises(self):
        row = dict(MOCK_GAIA_ROW_SOLAR)
        row["mass_flame"] = float("nan")
        with pytest.raises(ValueError, match="mass"):
            _extract_gaia("TIC_25155310", row, partial_tic8={})
    
    def test_masked_gaia_field_raises(self):
        """Masked Gaia catalog values are rejected before float conversion."""
        row = dict(MOCK_GAIA_ROW_SOLAR)
        row["mass_flame"] = np.ma.masked

        with pytest.raises(ValueError, match="mass"):
            _extract_gaia("TIC_25155310", row, partial_tic8={})

    def test_missing_gaia_percentile_raises(self):
        row = dict(MOCK_GAIA_ROW_SOLAR)
        row["mass_flame_lower"] = None
        with pytest.raises(ValueError, match="mass_flame_lower"):
            _extract_gaia("TIC_25155310", row, partial_tic8={})

    def test_tic8_values_take_priority_over_gaia(self):
        """
        Complete TIC-8 parameters are retained even when Gaia provides
        different values.
        """
        partial_tic8 = dict(MOCK_TIC8_ROW_SOLAR)

        params = _extract_gaia(
            "TIC_25155310",
            MOCK_GAIA_ROW_SOLAR,
            partial_tic8=partial_tic8,
        )

        assert params["mass"].to(u.M_sun).value == pytest.approx(1.03)
        assert params["radius"].to(u.R_sun).value == pytest.approx(1.13)
        assert params["teff"].to(u.K).value == pytest.approx(5765.0)
        assert params["luminosity"].to(u.L_sun).value == pytest.approx(1.35)

        assert params["catalog_source"] == "TIC-8+Gaia"


    def test_gaia_fills_missing_tic8_parameter(self):
        """
        Gaia supplies a parameter when TIC-8 has that parameter masked.
        Other complete TIC-8 parameters remain unchanged.
        """
        partial_tic8 = dict(MOCK_TIC8_ROW_SOLAR)
        partial_tic8["Mass"] = np.ma.masked
        partial_tic8["eneg_Mass"] = np.ma.masked
        partial_tic8["epos_Mass"] = np.ma.masked

        params = _extract_gaia(
            "TIC_25155310",
            MOCK_GAIA_ROW_SOLAR,
            partial_tic8=partial_tic8,
        )

        # Mass comes from Gaia.
        assert params["mass"].to(u.M_sun).value == pytest.approx(1.01)

        # Other parameters remain from TIC-8.
        assert params["radius"].to(u.R_sun).value == pytest.approx(1.13)
        assert params["teff"].to(u.K).value == pytest.approx(5765.0)
        assert params["luminosity"].to(u.L_sun).value == pytest.approx(1.35)

        assert params["catalog_source"] == "TIC-8+Gaia"


    def test_gaia_missing_field_raises_when_tic8_also_missing(self):
        """
        A parameter missing from both TIC-8 and Gaia must still fail.
        """
        partial_tic8 = dict(MOCK_TIC8_ROW_SOLAR)
        partial_tic8["Mass"] = np.ma.masked
        partial_tic8["eneg_Mass"] = np.ma.masked
        partial_tic8["epos_Mass"] = np.ma.masked

        gaia_row = dict(MOCK_GAIA_ROW_SOLAR)
        gaia_row["mass_flame"] = np.ma.masked

        with pytest.raises(ValueError, match="mass_flame"):
            _extract_gaia(
                "TIC_25155310",
                gaia_row,
                partial_tic8=partial_tic8,
            )


# ---------------------------------------------------------------------------
# 4. Cache behaviour
# ---------------------------------------------------------------------------

class TestCache:

    def test_save_and_load_round_trip(self, tmp_cache):
        """Saved params load back as identical values."""
        sp = make_stellar_params()
        params = {
            "tic_id": sp.tic_id,
            "catalog_source": sp.catalog_source,
            "catalog_version": None,
            "fetch_timestamp": datetime.now(timezone.utc),
            "mass": sp.mass, "mass_err_lo": sp.mass_err_lo, "mass_err_hi": sp.mass_err_hi,
            "radius": sp.radius, "radius_err_lo": sp.radius_err_lo, "radius_err_hi": sp.radius_err_hi,
            "teff": sp.teff, "teff_err_lo": sp.teff_err_lo, "teff_err_hi": sp.teff_err_hi,
            "luminosity": sp.luminosity, "luminosity_err_lo": sp.luminosity_err_lo,
            "luminosity_err_hi": sp.luminosity_err_hi,
        }
        save_to_cache(params, cache_dir=tmp_cache)
        loaded = load_from_cache(sp.tic_id, cache_dir=tmp_cache)
        assert loaded is not None
        assert loaded["mass"].to(u.M_sun).value == pytest.approx(
            sp.mass.to(u.M_sun).value, rel=1e-9
        )

    def test_cache_miss_returns_none(self, tmp_cache):
        result = load_from_cache("TIC_0000000", cache_dir=tmp_cache)
        assert result is None

    def test_invalidate_removes_entry(self, tmp_cache):
        sp = make_stellar_params()
        params = {
            "tic_id": sp.tic_id, "catalog_source": "TIC-8",
            "catalog_version": None,
            "fetch_timestamp": datetime.now(timezone.utc),
            "mass": sp.mass, "mass_err_lo": sp.mass_err_lo, "mass_err_hi": sp.mass_err_hi,
            "radius": sp.radius, "radius_err_lo": sp.radius_err_lo, "radius_err_hi": sp.radius_err_hi,
            "teff": sp.teff, "teff_err_lo": sp.teff_err_lo, "teff_err_hi": sp.teff_err_hi,
            "luminosity": sp.luminosity, "luminosity_err_lo": sp.luminosity_err_lo,
            "luminosity_err_hi": sp.luminosity_err_hi,
        }
        save_to_cache(params, cache_dir=tmp_cache)
        invalidate_cache(sp.tic_id, cache_dir=tmp_cache)
        assert load_from_cache(sp.tic_id, cache_dir=tmp_cache) is None

    def test_stale_cache_returns_none(self, tmp_cache):
        """Entry older than MAX_CACHE_AGE_DAYS treated as cache miss."""
        from utils.cache import MAX_CACHE_AGE_DAYS
        sp = make_stellar_params()
        old_ts = datetime.now(timezone.utc) - timedelta(days=MAX_CACHE_AGE_DAYS + 1)
        params = {
            "tic_id": sp.tic_id, "catalog_source": "TIC-8",
            "catalog_version": None,
            "fetch_timestamp": old_ts,
            "mass": sp.mass, "mass_err_lo": sp.mass_err_lo, "mass_err_hi": sp.mass_err_hi,
            "radius": sp.radius, "radius_err_lo": sp.radius_err_lo, "radius_err_hi": sp.radius_err_hi,
            "teff": sp.teff, "teff_err_lo": sp.teff_err_lo, "teff_err_hi": sp.teff_err_hi,
            "luminosity": sp.luminosity, "luminosity_err_lo": sp.luminosity_err_lo,
            "luminosity_err_hi": sp.luminosity_err_hi,
        }
        save_to_cache(params, cache_dir=tmp_cache)
        assert load_from_cache(sp.tic_id, cache_dir=tmp_cache) is None


# ---------------------------------------------------------------------------
# 5. fetch_stellar_params — end-to-end with mocked catalog calls
# ---------------------------------------------------------------------------

class TestFetchStellarParams:

    def _mock_query_tic8(self, row: dict):
        """Return a patch that makes _query_tic8 return the given row."""
        return patch("pce.fetcher._query_tic8", return_value=row)

    def test_returns_stellar_parameters_instance(self, tmp_cache):
        with self._mock_query_tic8(MOCK_TIC8_ROW_SOLAR):
            result = fetch_stellar_params("TIC_25155310", cache_dir=tmp_cache)
        assert isinstance(result, StellarParameters)

    def test_tic_id_normalised_on_return(self, tmp_cache):
        with self._mock_query_tic8(MOCK_TIC8_ROW_SOLAR):
            result = fetch_stellar_params("25155310", cache_dir=tmp_cache)
        assert result.tic_id == "TIC_25155310"

    def test_values_match_mock_row(self, tmp_cache):
        with self._mock_query_tic8(MOCK_TIC8_ROW_SOLAR):
            result = fetch_stellar_params("TIC_25155310", cache_dir=tmp_cache)
        assert result.mass.to(u.M_sun).value == pytest.approx(1.03, rel=1e-9)
        assert result.radius.to(u.R_sun).value == pytest.approx(1.13, rel=1e-9)

    def test_result_cached_after_fetch(self, tmp_cache):
        """Second call with same ID should hit cache (query called only once)."""
        with self._mock_query_tic8(MOCK_TIC8_ROW_SOLAR) as mock_q:
            fetch_stellar_params("TIC_25155310", cache_dir=tmp_cache)
            fetch_stellar_params("TIC_25155310", cache_dir=tmp_cache)
        mock_q.assert_called_once()

    def test_force_refetch_bypasses_cache(self, tmp_cache):
        """force_refetch=True should re-query even with a fresh cache entry."""
        with self._mock_query_tic8(MOCK_TIC8_ROW_SOLAR) as mock_q:
            fetch_stellar_params("TIC_25155310", cache_dir=tmp_cache)
            fetch_stellar_params("TIC_25155310", force_refetch=True, cache_dir=tmp_cache)
        assert mock_q.call_count == 2

    def test_negative_mass_raises_value_error(self, tmp_cache):
        with self._mock_query_tic8(MOCK_TIC8_ROW_BAD_NEGATIVE_MASS):
            with pytest.raises(ValueError):
                fetch_stellar_params("TIC_99999003", cache_dir=tmp_cache)

    def test_missing_uncertainties_raises_value_error(self, tmp_cache):
        """
        TIC-8 has NaN uncertainties → _extract_tic8 fails → fallback to Gaia.
        Gaia also has no entry → ValueError propagates to caller.
        """
        with self._mock_query_tic8(MOCK_TIC8_ROW_MISSING_UNCERTAINTIES), \
             patch("pce.fetcher._query_gaia", side_effect=ValueError("not in Gaia")):
            with pytest.raises((ValueError, RuntimeError)):
                fetch_stellar_params("TIC_99999004", cache_dir=tmp_cache)

    def test_gaia_fallback_on_incomplete_tic8(self, tmp_cache):
        """When TIC-8 is missing a field, Gaia fallback is used."""
        incomplete_row = dict(MOCK_TIC8_ROW_SOLAR)
        incomplete_row["lum"] = float("nan")   # force TIC-8 extraction to fail

        with patch("pce.fetcher._query_tic8", return_value=incomplete_row), \
             patch("pce.fetcher._query_gaia", return_value=MOCK_GAIA_ROW_SOLAR):
            result = fetch_stellar_params("TIC_25155310", cache_dir=tmp_cache)

        assert result.catalog_source == "TIC-8+Gaia"
        assert isinstance(result, StellarParameters)

    def test_mdwarf_params_correct(self, tmp_cache):
        with self._mock_query_tic8(MOCK_TIC8_ROW_MDWARF):
            result = fetch_stellar_params("TIC_99999002", cache_dir=tmp_cache)
        assert result.mass.to(u.M_sun).value == pytest.approx(0.35, rel=1e-9)
        assert result.teff.to(u.K).value == pytest.approx(3400.0, rel=1e-9)

    def test_asymmetric_errors_survive_round_trip(self, tmp_cache):
        """Asymmetric lo/hi uncertainties must be preserved through fetch + cache."""
        with self._mock_query_tic8(MOCK_TIC8_ROW_ASYMMETRIC):
            result = fetch_stellar_params("TIC_99999001", cache_dir=tmp_cache)
        assert result.mass_err_lo.to(u.M_sun).value == pytest.approx(0.08, rel=1e-6)
        assert result.mass_err_hi.to(u.M_sun).value == pytest.approx(0.15, rel=1e-6)

    def test_catalog_source_recorded(self, tmp_cache):
        with self._mock_query_tic8(MOCK_TIC8_ROW_SOLAR):
            result = fetch_stellar_params("TIC_25155310", cache_dir=tmp_cache)
        assert result.catalog_source == "TIC-8"

    def test_fetch_timestamp_is_recent(self, tmp_cache):
        with self._mock_query_tic8(MOCK_TIC8_ROW_SOLAR):
            result = fetch_stellar_params("TIC_25155310", cache_dir=tmp_cache)
        age = datetime.now(timezone.utc) - result.fetch_timestamp.replace(tzinfo=timezone.utc)
        assert age.total_seconds() < 10