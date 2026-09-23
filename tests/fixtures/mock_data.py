"""
mock_data.py — Shared Test Fixtures for PCE

Provides realistic mock data for unit and integration tests.
All values sourced from real catalog entries to ensure tests
reflect actual data shapes and edge cases.

Contents:
    MOCK_TIC8_RESPONSE_*   — astroquery TIC-8 table row dicts (real structure)
    MOCK_GAIA_RESPONSE_*   — astroquery Gaia table row dicts (real structure)
    MOCK_STELLAR_PARAMS_*  — pre-built StellarParameters instances
    CONFIRMED_PLANETS_*    — known planet parameters for validation tests
    make_stellar_params()  — factory for custom StellarParameters in tests
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timezone

import numpy as np
from astropy import units as u

from pce.schemas import StellarParameters


# ---------------------------------------------------------------------------
# TIC-8 raw table response mocks
# Mirrors the column structure returned by astroquery.mast.Catalogs.query_object
# with catalog="TIC". Only the columns PCE uses are included.
#
# Source star: TIC 25155310 (confirmed TESS host, well-characterised)
# Values from TIC-8 v8.2 via MAST
# ---------------------------------------------------------------------------

MOCK_TIC8_ROW_SOLAR = {
    # Identifiers
    "ID": "25155310",
    # Stellar parameters — real TIC-8 v8.2 column names
    # Source: TIC v8.2 schema (Paegert et al. 2021)
    "Mass":      1.03,
    "eneg_Mass": 0.07,
    "epos_Mass": 0.07,
    "Rad":       1.13,
    "eneg_Rad":  0.05,
    "epos_Rad":  0.05,
    "Teff":      5765.0,
    "eneg_Teff": 80.0,
    "epos_Teff": 80.0,
    "Lum":       1.35,
    "eneg_lum":  0.12,
    "epos_lum":  0.12,
    # Gaia source_id (TIC-8 "GAIA" column) — used for Gaia fallback
    "GAIA":      "3340074870763823104",
    "Version":   "8.2",
}

# Edge case: star with asymmetric uncertainties (hot subgiant)
MOCK_TIC8_ROW_ASYMMETRIC = {
    "ID":        "99999001",
    "Mass":      1.45,
    "eneg_Mass": 0.08,
    "epos_Mass": 0.15,      # noticeably asymmetric
    "Rad":       1.80,
    "eneg_Rad":  0.10,
    "epos_Rad":  0.20,
    "Teff":      6400.0,
    "eneg_Teff": 100.0,
    "epos_Teff": 150.0,
    "Lum":       4.20,
    "eneg_lum":  0.50,
    "epos_lum":  0.80,
    "GAIA":      "99999001000000000",
    "Version":   "8.2",
}

# Edge case: M-dwarf (small, cool, faint)
MOCK_TIC8_ROW_MDWARF = {
    "ID":        "99999002",
    "Mass":      0.35,
    "eneg_Mass": 0.03,
    "epos_Mass": 0.03,
    "Rad":       0.36,
    "eneg_Rad":  0.02,
    "epos_Rad":  0.02,
    "Teff":      3400.0,
    "eneg_Teff": 50.0,
    "epos_Teff": 50.0,
    "Lum":       0.025,
    "eneg_lum":  0.003,
    "epos_lum":  0.003,
    "GAIA":      "99999002000000000",
    "Version":   "8.2",
}

# Bad data: negative mass — should be rejected by validation
MOCK_TIC8_ROW_BAD_NEGATIVE_MASS = {
    "ID":        "99999003",
    "Mass":      -0.5,       # invalid
    "eneg_Mass": 0.05,
    "epos_Mass": 0.05,
    "Rad":       1.0,
    "eneg_Rad":  0.03,
    "epos_Rad":  0.03,
    "Teff":      5000.0,
    "eneg_Teff": 50.0,
    "epos_Teff": 50.0,
    "Lum":       0.8,
    "eneg_lum":  0.05,
    "epos_lum":  0.05,
    "GAIA":      "99999003000000000",
    "Version":   "8.2",
}

# Bad data: missing uncertainties (NaN) — should be rejected
MOCK_TIC8_ROW_MISSING_UNCERTAINTIES = {
    "ID":        "99999004",
    "Mass":      1.0,
    "eneg_Mass": float("nan"),   # missing
    "epos_Mass": float("nan"),   # missing
    "Rad":       1.0,
    "eneg_Rad":  float("nan"),
    "epos_Rad":  float("nan"),
    "Teff":      5778.0,
    "eneg_Teff": float("nan"),
    "epos_Teff": float("nan"),
    "Lum":       1.0,
    "eneg_lum":  float("nan"),
    "epos_lum":  float("nan"),
    "GAIA":      "99999004000000000",
    "Version":   "8.2",
}


# ---------------------------------------------------------------------------
# Gaia DR3 raw table response mocks
# Mirrors astroquery.gaia.Gaia.launch_job() result columns used by PCE.
# Gaia reports 16th/84th percentile bounds (not symmetric sigma).
#
# Source: Gaia DR3 astrophysical parameters for TIC 25155310 neighbourhood
# ---------------------------------------------------------------------------

MOCK_GAIA_ROW_SOLAR = {
    "source_id":            "3340074870763823104",
    # Real Gaia DR3 astrophysical_parameters column names
    "teff_gspphot":         5750.0,
    "teff_gspphot_lower":   5680.0,   # 16th percentile
    "teff_gspphot_upper":   5820.0,   # 84th percentile
    "radius_flame":         1.10,
    "radius_flame_lower":   1.04,
    "radius_flame_upper":   1.17,
    "mass_flame":           1.01,
    "mass_flame_lower":     0.92,
    "mass_flame_upper":     1.12,
    "lum_flame":            1.28,
    "lum_flame_lower":      1.15,
    "lum_flame_upper":      1.42,
}


# ---------------------------------------------------------------------------
# Pre-built StellarParameters instances (ready to use in tests)
# ---------------------------------------------------------------------------

def make_stellar_params(
    tic_id: str = "TIC_25155310",
    mass: float = 1.03,
    mass_err_lo: float = 0.07,
    mass_err_hi: float = 0.07,
    radius: float = 1.13,
    radius_err_lo: float = 0.05,
    radius_err_hi: float = 0.05,
    teff: float = 5765.0,
    teff_err_lo: float = 80.0,
    teff_err_hi: float = 80.0,
    luminosity: float = 1.35,
    luminosity_err_lo: float = 0.12,
    luminosity_err_hi: float = 0.12,
    catalog_source: str = "TIC-8",
) -> StellarParameters:
    """
    Factory for StellarParameters in tests.
    All values in solar units (M_sun, R_sun, K, L_sun).
    Defaults to TIC 25155310 — a well-characterised TESS host.
    Override individual fields as needed.
    """
    return StellarParameters(
        tic_id=tic_id,
        mass=mass * u.M_sun,
        mass_err_lo=mass_err_lo * u.M_sun,
        mass_err_hi=mass_err_hi * u.M_sun,
        radius=radius * u.R_sun,
        radius_err_lo=radius_err_lo * u.R_sun,
        radius_err_hi=radius_err_hi * u.R_sun,
        teff=teff * u.K,
        teff_err_lo=teff_err_lo * u.K,
        teff_err_hi=teff_err_hi * u.K,
        luminosity=luminosity * u.L_sun,
        luminosity_err_lo=luminosity_err_lo * u.L_sun,
        luminosity_err_hi=luminosity_err_hi * u.L_sun,
        catalog_source=catalog_source,
        fetch_timestamp=datetime.now(timezone.utc),
    )


# Pre-built instances for convenience
STELLAR_PARAMS_SOLAR = make_stellar_params()

STELLAR_PARAMS_MDWARF = make_stellar_params(
    tic_id="TIC_99999002",
    mass=0.35,       mass_err_lo=0.03,  mass_err_hi=0.03,
    radius=0.36,     radius_err_lo=0.02, radius_err_hi=0.02,
    teff=3400.0,     teff_err_lo=50.0,  teff_err_hi=50.0,
    luminosity=0.025, luminosity_err_lo=0.003, luminosity_err_hi=0.003,
)

STELLAR_PARAMS_ASYMMETRIC = make_stellar_params(
    tic_id="TIC_99999001",
    mass=1.45,       mass_err_lo=0.08,  mass_err_hi=0.15,
    radius=1.80,     radius_err_lo=0.10, radius_err_hi=0.20,
    teff=6400.0,     teff_err_lo=100.0, teff_err_hi=150.0,
    luminosity=4.20, luminosity_err_lo=0.50, luminosity_err_hi=0.80,
)


# ---------------------------------------------------------------------------
# Confirmed planet parameters for validation tests
# Used by validate_confirmed_planets.py and integration/test_pipeline.py
#
# Each entry: {star_id, period_day, duration_hr, depth}
# Source: NASA Exoplanet Archive (pscomppars table), confirmed planets
# ---------------------------------------------------------------------------

CONFIRMED_PLANETS = [
    {
        # WASP-17b — inflated hot Jupiter, well-characterised
        # Source: Anderson et al. (2010), ApJ, 709, 159
        # Depth = (R_p/R_star)^2 = (1.89 R_jup / 1.89 R_sun)^2 = 0.01056
        "star_id": "TIC_36734222",
        "planet_name": "WASP-17 b",
        "period_day": 3.7354,
        "duration_hr": 4.35,
        "depth": 0.01056,
    },
    {
    # TOI-132 b — sub-Neptune, TESS discovery
    # Source: Díaz et al. (2020), MNRAS, 493, 973
    "star_id": "TIC_89020549",
    "planet_name": "TOI-132 b",
    "period_day": 2.1097,
    "duration_hr": 2.63,
    "depth": 0.00121,
    },
    {
        # HD 209458 b — first transiting planet discovered, benchmark
        # Source: Charbonneau et al. (2000), ApJ, 529, L45
        "star_id": "TIC_420814525",
        "planet_name": "HD 209458 b",
        "period_day": 3.5247,
        "duration_hr": 2.98,
        "depth": 0.01474,
    },
    {
        # L 98-59 b — small rocky planet around M-dwarf
        # Source: Cloutier et al. (2019), A&A, 629, A111
        "star_id": "TIC_307210830",
        "planet_name": "L 98-59 b",
        "period_day": 2.2531,
        "duration_hr": 0.79,
        "depth": 0.00049,
    },
    {
        # TRAPPIST-1 b — ultra-cool dwarf system benchmark
        # Source: Gillon et al. (2017), Nature, 542, 456
        "star_id": "TIC_278956474",
        "planet_name": "TRAPPIST-1 b",
        "period_day": 1.5109,
        "duration_hr": 0.60,
        "depth": 0.00726,
    },
]
