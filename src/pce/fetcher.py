"""
fetcher.py — Stellar Parameter Fetcher (Stage 2)

Queries TIC-8 via astroquery, falls back to Gaia DR3 if TIC-8 data
is incomplete, validates all parameters, caches to CSV, and returns
a StellarParameters Pydantic model.

Public API:
    fetch_stellar_params(star_id, force_refetch=False) -> StellarParameters

Internal flow:
    1. Check CSV cache — return immediately if fresh entry exists
    2. Query TIC-8 via astroquery.mast.Catalogs
    3. If TIC-8 is missing any required field, fall back to Gaia DR3
    4. Validate all parameters (validation.py)
    5. Cache result to CSV
    6. Return StellarParameters

Error contract:
    ValueError  — missing/invalid data, star not found, validation failure
    RuntimeError — unexpected astroquery failure
"""

import math
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
from typing import Optional

from astropy import units as u
from astropy.units import Quantity

from pce.schemas import StellarParameters
from utils.cache import load_from_cache, save_to_cache
from utils.validation import validate_all_stellar_params


# ---------------------------------------------------------------------------
# Cache directory (default; overrideable in tests)
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_stellar_params(
    star_id: str,
    force_refetch: bool = False,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> StellarParameters:
    """
    Fetch, validate, and return stellar parameters for star_id.

    Checks the CSV cache first. On a miss (or if force_refetch=True),
    queries TIC-8 and falls back to Gaia DR3 for any missing fields.

    Args:
        star_id:       TIC identifier, e.g. "TIC_25155310" or "25155310"
        force_refetch: If True, bypass cache and re-query the catalog
        cache_dir:     Directory for CSV cache (default: data/cache/)

    Returns:
        StellarParameters — validated Pydantic model with Quantity fields

    Raises:
        ValueError:  star not found, required parameters missing or invalid
        RuntimeError: unexpected astroquery failure
    """
    tic_id = _normalise_tic_id(star_id)

    # ------------------------------------------------------------------
    # 1. Cache check
    # ------------------------------------------------------------------
    if not force_refetch:
        cached = load_from_cache(tic_id, cache_dir=cache_dir)
        if cached is not None:
            return StellarParameters(**cached)

    # ------------------------------------------------------------------
    # 2. Query TIC-8
    # ------------------------------------------------------------------
    raw = _query_tic8(tic_id)

    # ------------------------------------------------------------------
    # 3. Extract parameters; fall back to Gaia for missing fields
    # ------------------------------------------------------------------
    try:
        params = _extract_tic8(tic_id, raw)
    except ValueError as exc:
        # TIC-8 incomplete — attempt Gaia fallback using Gaia source_id from TIC-8
        gaia_source_id = raw.get("GAIA", None)
        if not gaia_source_id or str(gaia_source_id).strip() in ("", "0", "None"):
            raise ValueError(
                f"TIC-8 parameters incomplete for {tic_id} and no Gaia source_id "
                f"available for fallback. Original error: {exc}"
            ) from exc
        gaia_raw = _query_gaia(tic_id, gaia_source_id=str(gaia_source_id))
        params = _extract_gaia(tic_id, gaia_raw, partial_tic8=raw)

    # ------------------------------------------------------------------
    # 4. Validate
    # ------------------------------------------------------------------
    validate_all_stellar_params(
        mass=params["mass"],
        radius=params["radius"],
        teff=params["teff"],
        luminosity=params["luminosity"],
        mass_err_lo=params["mass_err_lo"],
        mass_err_hi=params["mass_err_hi"],
        radius_err_lo=params["radius_err_lo"],
        radius_err_hi=params["radius_err_hi"],
        teff_err_lo=params["teff_err_lo"],
        teff_err_hi=params["teff_err_hi"],
        luminosity_err_lo=params["luminosity_err_lo"],
        luminosity_err_hi=params["luminosity_err_hi"],
    )

    # ------------------------------------------------------------------
    # 5. Cache
    # ------------------------------------------------------------------
    save_to_cache(params, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # 6. Return
    # ------------------------------------------------------------------
    return StellarParameters(**params)


# ---------------------------------------------------------------------------
# TIC-8 query
# ---------------------------------------------------------------------------

def _normalise_tic_id(star_id: str) -> str:
    """
    Normalise star_id to "TIC_<number>" form.
    Accepts: "TIC_25155310", "TIC25155310", "25155310"
    """
    s = star_id.strip().upper().replace("TIC_", "").replace("TIC", "")
    if not s.isdigit():
        raise ValueError(
            f"Cannot parse star_id '{star_id}' as a TIC identifier. "
            f"Expected e.g. 'TIC_25155310' or '25155310'."
        )
    return f"TIC_{s}"


def _query_tic8(tic_id: str) -> dict:
    """
    Query TIC-8 via astroquery.mast.Catalogs.
    Returns raw row dict.

    Raises:
        ValueError:  star not found in TIC-8
        RuntimeError: astroquery failure
    """
    try:
        from astroquery.mast import Catalogs
    except ImportError as e:
        raise RuntimeError(
            "astroquery is not installed. Run: pip install astroquery"
        ) from e

    numeric_id = int(tic_id.replace("TIC_", ""))

    try:
        result = Catalogs.query_criteria(
            catalog="Tic",
            ID=numeric_id,
        )
    except Exception as exc:
        raise RuntimeError(f"astroquery TIC-8 query failed for {tic_id}: {exc}") from exc

    if result is None or len(result) == 0:
        raise ValueError(f"Star {tic_id} not found in TIC-8.")

    # astroquery returns an astropy Table — take first row as dict
    return {col: result[col][0] for col in result.colnames}

def _is_missing_catalog_value(value: object) -> bool:
    """
    Return True when a catalog value is missing, masked, or non-finite.

    Handles:
        - None
        - Astropy/NumPy masked values
        - NaN
        - positive/negative infinity
        - non-numeric values
    """
    if value is None:
        return True

    if np.ma.is_masked(value):
        return True

    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True

def _extract_tic8(tic_id: str, row: dict) -> dict:
    """
    Extract and unit-attach stellar parameters from a TIC-8 row dict.

    TIC-8 uncertainty convention:
        eneg_<param>  — lower uncertainty (positive value)
        epos_<param>  — upper uncertainty (positive value)

    Returns a flat dict with Quantity values ready for StellarParameters.

    Raises:
        ValueError: if any required field is missing or NaN
    """
    def _get(key: str, unit: u.Unit, label: str) -> Quantity:
        val = row.get(key)

        if _is_missing_catalog_value(val):
            raise ValueError(
                f"TIC-8 field '{key}' is missing or non-finite for {tic_id} "
                f"(required for {label})"
            )

        return float(val) * unit

    # Real TIC-8 v8.2 column names (from official schema):
    #   Mass / eneg_Mass / epos_Mass  (capital M)
    #   Rad  / eneg_Rad  / epos_Rad   (capital R)
    #   Teff / eneg_Teff / epos_Teff  (capital T)
    #   Lum  / eneg_lum  / epos_lum   (mixed case)
    #   Version (capital V)
    mass         = _get("Mass",      u.M_sun, "mass")
    mass_err_lo  = _get("eneg_Mass", u.M_sun, "mass lower uncertainty")
    mass_err_hi  = _get("epos_Mass", u.M_sun, "mass upper uncertainty")

    radius        = _get("Rad",      u.R_sun, "radius")
    radius_err_lo = _get("eneg_Rad", u.R_sun, "radius lower uncertainty")
    radius_err_hi = _get("epos_Rad", u.R_sun, "radius upper uncertainty")

    teff        = _get("Teff",      u.K, "Teff")
    teff_err_lo = _get("eneg_Teff", u.K, "Teff lower uncertainty")
    teff_err_hi = _get("epos_Teff", u.K, "Teff upper uncertainty")

    luminosity        = _get("Lum",      u.L_sun, "luminosity")
    luminosity_err_lo = _get("eneg_lum", u.L_sun, "luminosity lower uncertainty")
    luminosity_err_hi = _get("epos_lum", u.L_sun, "luminosity upper uncertainty")

    return {
        "tic_id": tic_id,
        "catalog_source": "TIC-8",
        "catalog_version": str(row.get("Version", "")),
        "fetch_timestamp": datetime.now(timezone.utc),
        "mass": mass,                         "mass_err_lo": mass_err_lo,
        "mass_err_hi": mass_err_hi,
        "radius": radius,                     "radius_err_lo": radius_err_lo,
        "radius_err_hi": radius_err_hi,
        "teff": teff,                         "teff_err_lo": teff_err_lo,
        "teff_err_hi": teff_err_hi,
        "luminosity": luminosity,             "luminosity_err_lo": luminosity_err_lo,
        "luminosity_err_hi": luminosity_err_hi,
    }


# ---------------------------------------------------------------------------
# Gaia DR3 fallback
# ---------------------------------------------------------------------------

def _query_gaia(tic_id: str, gaia_source_id: str) -> dict:
    """
    Query Gaia DR3 astrophysical parameters using a Gaia DR3 source_id.

    Args:
        tic_id:         TIC identifier (for error messages only)
        gaia_source_id: Gaia DR3 source_id from TIC-8's "GAIA" column

    Raises:
        ValueError:  star not found in Gaia DR3
        RuntimeError: astroquery failure
    """
    try:
        from astroquery.gaia import Gaia
    except ImportError as e:
        raise RuntimeError(
            "astroquery is not installed. Run: pip install astroquery"
        ) from e

    numeric_id = str(gaia_source_id).strip()

    # Strategy: caller passes the Gaia DR3 source_id extracted from
    # TIC-8's "GAIA" column. We query gaiadr3.astrophysical_parameters directly.
    # Real Gaia DR3 column names (from official schema):
    #   teff_gspphot / teff_gspphot_lower / teff_gspphot_upper
    #   radius_flame / radius_flame_lower / radius_flame_upper
    #   mass_flame   / mass_flame_lower   / mass_flame_upper
    #   lum_flame    / lum_flame_lower    / lum_flame_upper
    adql = f"""
        SELECT
            source_id,
            teff_gspphot, teff_gspphot_lower, teff_gspphot_upper,
            radius_flame, radius_flame_lower, radius_flame_upper,
            mass_flame, mass_flame_lower, mass_flame_upper,
            lum_flame, lum_flame_lower, lum_flame_upper
        FROM gaiadr3.astrophysical_parameters
        WHERE source_id = {numeric_id}
    """

    try:
        job = Gaia.launch_job(adql)
        result = job.get_results()
    except Exception as exc:
        raise RuntimeError(
            f"Gaia DR3 query failed for {tic_id}: {exc}"
        ) from exc

    if result is None or len(result) == 0:
        raise ValueError(
            f"Star {tic_id} not found in Gaia DR3 astrophysical parameters. "
            f"Cannot complete parameter set."
        )

    return dict(zip(result.colnames, result[0]))

def _extract_gaia(tic_id: str, gaia_row: dict, partial_tic8: dict) -> dict:
    """
    Extract stellar parameters using Gaia DR3 to complete an incomplete
    TIC-8 parameter set.

    TIC-8 values take priority where the complete central value and both
    uncertainties are present and valid. Gaia DR3 supplies only the fields
    that TIC-8 cannot provide.

    Gaia uncertainty convention:
        <param>_lower  — 16th percentile value
        <param>_upper  — 84th percentile value

    We convert 16th/84th percentiles to asymmetric uncertainties:
        err_lo = central - p16
        err_hi = p84 - central

    Raises:
        ValueError: if any required field is missing from both catalogs
    """

    def _tic8_value_and_errors(
        val_key: str,
        lo_key: str,
        hi_key: str,
        unit: u.Unit,
        label: str,
    ):
        """
        Return a complete TIC-8 parameter triplet if all three values
        are present and valid. Otherwise return None.
        """
        val = partial_tic8.get(val_key)
        lo = partial_tic8.get(lo_key)
        hi = partial_tic8.get(hi_key)

        if (
            _is_missing_catalog_value(val)
            or _is_missing_catalog_value(lo)
            or _is_missing_catalog_value(hi)
        ):
            return None

        return (
            float(val) * unit,
            float(lo) * unit,
            float(hi) * unit,
        )

    def _gaia_value_and_errors(
        val_key: str,
        lo_key: str,
        hi_key: str,
        unit: u.Unit,
        label: str,
    ):
        """
        Extract central + asymmetric errors from Gaia percentile columns.
        """
        val = gaia_row.get(val_key)
        lo = gaia_row.get(lo_key)
        hi = gaia_row.get(hi_key)

        for name, v in [(val_key, val), (lo_key, lo), (hi_key, hi)]:
            if _is_missing_catalog_value(v):
                raise ValueError(
                    f"Gaia DR3 field '{name}' is missing or non-finite for "
                    f"{tic_id} (required for {label})"
                )

        central = float(val)
        err_lo = max(central - float(lo), 0.0)
        err_hi = max(float(hi) - central, 0.0)

        return (
            central * unit,
            err_lo * unit,
            err_hi * unit,
        )

    def _resolve_parameter(
        tic_val_key: str,
        tic_lo_key: str,
        tic_hi_key: str,
        gaia_val_key: str,
        gaia_lo_key: str,
        gaia_hi_key: str,
        unit: u.Unit,
        label: str,
    ):
        """
        Prefer a complete TIC-8 parameter; otherwise use Gaia DR3.
        """
        tic_values = _tic8_value_and_errors(
            tic_val_key,
            tic_lo_key,
            tic_hi_key,
            unit,
            label,
        )

        if tic_values is not None:
            return tic_values

        return _gaia_value_and_errors(
            gaia_val_key,
            gaia_lo_key,
            gaia_hi_key,
            unit,
            label,
        )

    mass, mass_err_lo, mass_err_hi = _resolve_parameter(
        "Mass",
        "eneg_Mass",
        "epos_Mass",
        "mass_flame",
        "mass_flame_lower",
        "mass_flame_upper",
        u.M_sun,
        "mass",
    )

    radius, radius_err_lo, radius_err_hi = _resolve_parameter(
        "Rad",
        "eneg_Rad",
        "epos_Rad",
        "radius_flame",
        "radius_flame_lower",
        "radius_flame_upper",
        u.R_sun,
        "radius",
    )

    teff, teff_err_lo, teff_err_hi = _resolve_parameter(
        "Teff",
        "eneg_Teff",
        "epos_Teff",
        "teff_gspphot",
        "teff_gspphot_lower",
        "teff_gspphot_upper",
        u.K,
        "Teff",
    )

    luminosity, luminosity_err_lo, luminosity_err_hi = _resolve_parameter(
        "Lum",
        "eneg_lum",
        "epos_lum",
        "lum_flame",
        "lum_flame_lower",
        "lum_flame_upper",
        u.L_sun,
        "luminosity",
    )

    return {
        "tic_id": tic_id,
        "catalog_source": "TIC-8+Gaia",
        "catalog_version": "TIC-8 + DR3",
        "fetch_timestamp": datetime.now(timezone.utc),

        "mass": mass,
        "mass_err_lo": mass_err_lo,
        "mass_err_hi": mass_err_hi,

        "radius": radius,
        "radius_err_lo": radius_err_lo,
        "radius_err_hi": radius_err_hi,

        "teff": teff,
        "teff_err_lo": teff_err_lo,
        "teff_err_hi": teff_err_hi,

        "luminosity": luminosity,
        "luminosity_err_lo": luminosity_err_lo,
        "luminosity_err_hi": luminosity_err_hi,
    }