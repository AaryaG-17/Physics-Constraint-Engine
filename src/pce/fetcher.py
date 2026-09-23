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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
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
        # TIC-8 incomplete — attempt Gaia fallback using Gaia source_id
        # from TIC-8.
        gaia_source_id = raw.get("GAIA", None)

        if not gaia_source_id or str(gaia_source_id).strip() in (
            "",
            "0",
            "None",
        ):
            raise ValueError(
                f"TIC-8 parameters incomplete for {tic_id} and no Gaia "
                f"source_id available for fallback. Original error: {exc}"
            ) from exc

        gaia_raw = _query_gaia(
            tic_id,
            gaia_source_id=str(gaia_source_id),
        )

        params = _extract_gaia(
            tic_id,
            gaia_raw,
            partial_tic8=raw,
        )

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

    Accepts:
        "TIC_25155310"
        "TIC25155310"
        "25155310"
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

    Retries transient astroquery/MAST failures up to three times.

    Returns:
        Raw TIC-8 row as a dictionary.

    Raises:
        ValueError:  star not found in TIC-8
        RuntimeError: astroquery failure after retries
    """
    try:
        from astroquery.mast import Catalogs
    except ImportError as exc:
        raise RuntimeError(
            "astroquery is not installed. Run: pip install astroquery"
        ) from exc

    numeric_id = int(tic_id.replace("TIC_", ""))

    last_error = None

    for attempt in range(3):
        try:
            result = Catalogs.query_criteria(
                catalog="Tic",
                ID=numeric_id,
            )

            if result is None or len(result) == 0:
                raise ValueError(
                    f"Star {tic_id} not found in TIC-8."
                )

            # astroquery returns an astropy Table — take first row as dict.
            return {
                col: result[col][0]
                for col in result.colnames
            }

        except ValueError:
            # A genuine "not found" result should not be retried.
            raise

        except Exception as exc:
            last_error = exc

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"astroquery TIC-8 query failed for {tic_id}: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Catalog value validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TIC-8 extraction
# ---------------------------------------------------------------------------

def _extract_tic8(tic_id: str, raw: dict) -> dict:
    """
    Extract stellar parameters from a TIC-8 row.

    TIC-8 central-value fields:
        mass
        rad
        Teff
        lum

    TIC-8 uncertainty priority:
        1. Asymmetric uncertainties: eneg_* / epos_*
        2. Symmetric uncertainty: e_*
        3. If neither is available, raise ValueError so that Gaia fallback
           can be attempted.
    """

    def _get_required(
        name: str,
        unit: u.Unit,
        label: str,
    ) -> Quantity:
        value = raw.get(name)

        if _is_missing_catalog_value(value):
            raise ValueError(
                f"TIC-8 field '{name}' is missing or non-finite for "
                f"{tic_id} (required for {label})"
            )

        return float(value) * unit

    def _get_uncertainty(
        parameter: str,
        symmetric_parameter: str,
        label: str,
        unit: u.Unit,
    ) -> tuple[Quantity, Quantity]:
        """
        Return (lower_error, upper_error).

        Prefer asymmetric TIC uncertainties. If unavailable, use the
        symmetric e_* uncertainty for both sides.
        """

        eneg_name = f"eneg_{parameter}"
        epos_name = f"epos_{parameter}"
        symmetric_name = f"e_{symmetric_parameter}"

        eneg = raw.get(eneg_name)
        epos = raw.get(epos_name)

        # --------------------------------------------------------------
        # 1. Prefer asymmetric uncertainties.
        # --------------------------------------------------------------
        if (
            not _is_missing_catalog_value(eneg)
            and not _is_missing_catalog_value(epos)
        ):
            return (
                abs(float(eneg)) * unit,
                abs(float(epos)) * unit,
            )

        # --------------------------------------------------------------
        # 2. Fall back to symmetric uncertainty.
        # --------------------------------------------------------------
        symmetric = raw.get(symmetric_name)

        if not _is_missing_catalog_value(symmetric):
            error = abs(float(symmetric)) * unit
            return error, error

        # --------------------------------------------------------------
        # 3. Neither uncertainty is usable.
        # --------------------------------------------------------------
        raise ValueError(
            f"TIC-8 uncertainty for '{label}' is missing or non-finite "
            f"for {tic_id}"
        )

    # ------------------------------------------------------------------
    # Central values — LIVE TIC-8 field names
    # ------------------------------------------------------------------

    mass = _get_required(
        "mass",
        u.M_sun,
        "mass",
    )

    radius = _get_required(
        "rad",
        u.R_sun,
        "radius",
    )

    teff = _get_required(
        "Teff",
        u.K,
        "Teff",
    )

    luminosity = _get_required(
        "lum",
        u.L_sun,
        "luminosity",
    )

    # ------------------------------------------------------------------
    # Uncertainties
    #
    # Asymmetric → symmetric fallback.
    # ------------------------------------------------------------------

    mass_err_lo, mass_err_hi = _get_uncertainty(
        "Mass",
        "mass",
        "mass",
        u.M_sun,
    )

    radius_err_lo, radius_err_hi = _get_uncertainty(
        "Rad",
        "rad",
        "radius",
        u.R_sun,
    )

    teff_err_lo, teff_err_hi = _get_uncertainty(
        "Teff",
        "Teff",
        "Teff",
        u.K,
    )

    luminosity_err_lo, luminosity_err_hi = _get_uncertainty(
        "Lum",
        "lum",
        "luminosity",
        u.L_sun,
    )

    return {
        "tic_id": tic_id,
        "catalog_source": "TIC-8",
        "catalog_version": "TIC-8",
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


# ---------------------------------------------------------------------------
# Gaia DR3 fallback
# ---------------------------------------------------------------------------

def _query_gaia(tic_id: str, gaia_source_id: str) -> dict:
    """
    Query Gaia DR3 astrophysical parameters using a Gaia DR3 source_id.

    Args:
        tic_id:       TIC identifier (for error messages only)
        gaia_source_id:
            Gaia DR3 source_id from TIC-8's "GAIA" column

    Raises:
        ValueError:  star not found in Gaia DR3
        RuntimeError: astroquery failure
    """
    try:
        from astroquery.gaia import Gaia
    except ImportError as exc:
        raise RuntimeError(
            "astroquery is not installed. Run: pip install astroquery"
        ) from exc

    numeric_id = str(gaia_source_id).strip()

    # Gaia DR3 astrophysical_parameters columns:
    #
    # teff_gspphot / teff_gspphot_lower / teff_gspphot_upper
    # radius_flame / radius_flame_lower / radius_flame_upper
    # mass_flame   / mass_flame_lower   / mass_flame_upper
    # lum_flame    / lum_flame_lower    / lum_flame_upper

    adql = f"""
        SELECT
            source_id,
            teff_gspphot,
            teff_gspphot_lower,
            teff_gspphot_upper,
            radius_flame,
            radius_flame_lower,
            radius_flame_upper,
            mass_flame,
            mass_flame_lower,
            mass_flame_upper,
            lum_flame,
            lum_flame_lower,
            lum_flame_upper
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
            f"Star {tic_id} not found in Gaia DR3 astrophysical "
            f"parameters. Cannot complete parameter set."
        )

    return dict(zip(result.colnames, result[0]))


def _extract_gaia(
    tic_id: str,
    gaia_row: dict,
    partial_tic8: dict,
) -> dict:
    """
    Extract stellar parameters using Gaia DR3 to complete an incomplete
    TIC-8 parameter set.

    TIC-8 values take priority where a complete central value and
    uncertainty set is present and valid. Gaia DR3 supplies fields
    that TIC-8 cannot provide.

    TIC-8 uncertainty priority:
        1. Asymmetric eneg_*/epos_*
        2. Symmetric e_*

    Gaia uncertainty convention:
        <param>_lower — 16th percentile value
        <param>_upper — 84th percentile value

    Gaia uncertainties are converted to:

        err_lo = central - p16
        err_hi = p84 - central

    Raises:
        ValueError: if any required field is missing from both catalogs
    """

    def _tic8_value_and_errors(
        value_keys: tuple[str, ...],
        asymmetric_keys: tuple[str, str],
        symmetric_key: str,
        unit: u.Unit,
    ):
        """
        Return TIC-8 central value and asymmetric uncertainties.

        Asymmetric uncertainty is preferred. Symmetric uncertainty is
        used as fallback.
        """

        value = None

        for key in value_keys:
            candidate = partial_tic8.get(key)

            if not _is_missing_catalog_value(candidate):
                value = candidate
                break

        if _is_missing_catalog_value(value):
            return None

        eneg = partial_tic8.get(asymmetric_keys[0])
        epos = partial_tic8.get(asymmetric_keys[1])

        # Prefer asymmetric uncertainties.
        if (
            not _is_missing_catalog_value(eneg)
            and not _is_missing_catalog_value(epos)
        ):
            return (
                float(value) * unit,
                abs(float(eneg)) * unit,
                abs(float(epos)) * unit,
            )

        # Fall back to symmetric uncertainty.
        symmetric = partial_tic8.get(symmetric_key)

        if not _is_missing_catalog_value(symmetric):
            error = abs(float(symmetric)) * unit

            return (
                float(value) * unit,
                error,
                error,
            )

        return None

    def _gaia_value_and_errors(
        val_key: str,
        lo_key: str,
        hi_key: str,
        unit: u.Unit,
        label: str,
    ):
        """
        Extract central value and asymmetric errors from Gaia percentile
        columns.
        """

        val = gaia_row.get(val_key)
        lo = gaia_row.get(lo_key)
        hi = gaia_row.get(hi_key)

        for name, value in (
            (val_key, val),
            (lo_key, lo),
            (hi_key, hi),
        ):
            if _is_missing_catalog_value(value):
                raise ValueError(
                    f"Gaia DR3 field '{name}' is missing or non-finite "
                    f"for {tic_id} (required for {label})"
                )

        central = float(val)

        err_lo = max(
            central - float(lo),
            0.0,
        )

        err_hi = max(
            float(hi) - central,
            0.0,
        )

        return (
            central * unit,
            err_lo * unit,
            err_hi * unit,
        )

    def _resolve_parameter(
        tic_value_keys: tuple[str, ...],
        tic_asymmetric_keys: tuple[str, str],
        tic_symmetric_key: str,
        gaia_val_key: str,
        gaia_lo_key: str,
        gaia_hi_key: str,
        unit: u.Unit,
        label: str,
    ):
        """
        Prefer TIC-8 when a complete value + uncertainty set exists.
        Otherwise use Gaia DR3.
        """

        tic_values = _tic8_value_and_errors(
            value_keys=tic_value_keys,
            asymmetric_keys=tic_asymmetric_keys,
            symmetric_key=tic_symmetric_key,
            unit=unit,
        )

        if tic_values is not None:
            return tic_values

        return _gaia_value_and_errors(
            val_key=gaia_val_key,
            lo_key=gaia_lo_key,
            hi_key=gaia_hi_key,
            unit=unit,
            label=label,
        )

    # ------------------------------------------------------------------
    # Mass
    # ------------------------------------------------------------------

    mass, mass_err_lo, mass_err_hi = _resolve_parameter(
        tic_value_keys=("mass",),
        tic_asymmetric_keys=("eneg_Mass", "epos_Mass"),
        tic_symmetric_key="e_mass",
        gaia_val_key="mass_flame",
        gaia_lo_key="mass_flame_lower",
        gaia_hi_key="mass_flame_upper",
        unit=u.M_sun,
        label="mass",
    )

    # ------------------------------------------------------------------
    # Radius
    # ------------------------------------------------------------------

    radius, radius_err_lo, radius_err_hi = _resolve_parameter(
        tic_value_keys=("rad",),
        tic_asymmetric_keys=("eneg_Rad", "epos_Rad"),
        tic_symmetric_key="e_rad",
        gaia_val_key="radius_flame",
        gaia_lo_key="radius_flame_lower",
        gaia_hi_key="radius_flame_upper",
        unit=u.R_sun,
        label="radius",
    )

    # ------------------------------------------------------------------
    # Effective temperature
    # ------------------------------------------------------------------

    teff, teff_err_lo, teff_err_hi = _resolve_parameter(
        tic_value_keys=("Teff",),
        tic_asymmetric_keys=("eneg_Teff", "epos_Teff"),
        tic_symmetric_key="e_Teff",
        gaia_val_key="teff_gspphot",
        gaia_lo_key="teff_gspphot_lower",
        gaia_hi_key="teff_gspphot_upper",
        unit=u.K,
        label="Teff",
    )

    # ------------------------------------------------------------------
    # Luminosity
    # ------------------------------------------------------------------

    luminosity, luminosity_err_lo, luminosity_err_hi = _resolve_parameter(
        tic_value_keys=("lum",),
        tic_asymmetric_keys=("eneg_Lum", "epos_Lum"),
        tic_symmetric_key="e_lum",
        gaia_val_key="lum_flame",
        gaia_lo_key="lum_flame_lower",
        gaia_hi_key="lum_flame_upper",
        unit=u.L_sun,
        label="luminosity",
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