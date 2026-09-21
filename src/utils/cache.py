"""
cache.py — Stellar Parameter CSV Cache

Saves and loads fetched stellar parameters to/from local CSV files
so repeated runs on the same star avoid redundant network calls.

Cache file naming:
    data/cache/<tic_id>.csv

Each CSV stores one row of parameters + metadata (timestamp, catalog source).
The fetcher checks the cache before querying TIC-8 or Gaia.

Cache invalidation:
    - Explicit: pass force_refetch=True to fetch_stellar_params()
    - By age:   entries older than MAX_CACHE_AGE_DAYS are treated as stale
"""

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from astropy import units as u
from astropy.units import Quantity

# ---------------------------------------------------------------------------
# Cache location — relative to project root, created at runtime if absent
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Cache entries older than this are considered stale
MAX_CACHE_AGE_DAYS = 30

# CSV column order (must match _to_row / _from_row exactly)
_COLUMNS = [
    "tic_id",
    "catalog_source",
    "catalog_version",
    "fetch_timestamp",
    "mass_Msun",
    "mass_err_lo_Msun",
    "mass_err_hi_Msun",
    "radius_Rsun",
    "radius_err_lo_Rsun",
    "radius_err_hi_Rsun",
    "teff_K",
    "teff_err_lo_K",
    "teff_err_hi_K",
    "luminosity_Lsun",
    "luminosity_err_lo_Lsun",
    "luminosity_err_hi_Lsun",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(tic_id: str, cache_dir: Path) -> Path:
    safe_id = tic_id.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_id}.csv"


def _to_row(params: dict) -> dict:
    """
    Flatten a dict of Quantity stellar params into plain floats for CSV storage.
    params keys must match _COLUMNS (without tic_id / metadata fields).
    """
    return {
        "tic_id": params["tic_id"],
        "catalog_source": params["catalog_source"],
        "catalog_version": params.get("catalog_version", ""),
        "fetch_timestamp": params["fetch_timestamp"],
        "mass_Msun": params["mass"].to(u.M_sun).value,
        "mass_err_lo_Msun": params["mass_err_lo"].to(u.M_sun).value,
        "mass_err_hi_Msun": params["mass_err_hi"].to(u.M_sun).value,
        "radius_Rsun": params["radius"].to(u.R_sun).value,
        "radius_err_lo_Rsun": params["radius_err_lo"].to(u.R_sun).value,
        "radius_err_hi_Rsun": params["radius_err_hi"].to(u.R_sun).value,
        "teff_K": params["teff"].to(u.K).value,
        "teff_err_lo_K": params["teff_err_lo"].to(u.K).value,
        "teff_err_hi_K": params["teff_err_hi"].to(u.K).value,
        "luminosity_Lsun": params["luminosity"].to(u.L_sun).value,
        "luminosity_err_lo_Lsun": params["luminosity_err_lo"].to(u.L_sun).value,
        "luminosity_err_hi_Lsun": params["luminosity_err_hi"].to(u.L_sun).value,
    }


def _from_row(row: dict) -> dict:
    """Reconstruct Quantity stellar params from a CSV row dict."""
    return {
        "tic_id": row["tic_id"],
        "catalog_source": row["catalog_source"],
        "catalog_version": row["catalog_version"] or None,
        "fetch_timestamp": datetime.fromisoformat(row["fetch_timestamp"]),
        "mass": float(row["mass_Msun"]) * u.M_sun,
        "mass_err_lo": float(row["mass_err_lo_Msun"]) * u.M_sun,
        "mass_err_hi": float(row["mass_err_hi_Msun"]) * u.M_sun,
        "radius": float(row["radius_Rsun"]) * u.R_sun,
        "radius_err_lo": float(row["radius_err_lo_Rsun"]) * u.R_sun,
        "radius_err_hi": float(row["radius_err_hi_Rsun"]) * u.R_sun,
        "teff": float(row["teff_K"]) * u.K,
        "teff_err_lo": float(row["teff_err_lo_K"]) * u.K,
        "teff_err_hi": float(row["teff_err_hi_K"]) * u.K,
        "luminosity": float(row["luminosity_Lsun"]) * u.L_sun,
        "luminosity_err_lo": float(row["luminosity_err_lo_Lsun"]) * u.L_sun,
        "luminosity_err_hi": float(row["luminosity_err_hi_Lsun"]) * u.L_sun,
    }


def _is_stale(fetch_timestamp: datetime) -> bool:
    """Return True if the cache entry is older than MAX_CACHE_AGE_DAYS."""
    if fetch_timestamp.tzinfo is None:
        fetch_timestamp = fetch_timestamp.replace(tzinfo=timezone.utc)
    age = datetime.now(tz=timezone.utc) - fetch_timestamp
    return age > timedelta(days=MAX_CACHE_AGE_DAYS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_from_cache(
    tic_id: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> Optional[dict]:
    """
    Load stellar parameters from cache for tic_id.

    Returns:
        dict of Quantity parameters if a fresh cache entry exists, else None.
        Returns None if the file doesn't exist or the entry is stale.
    """
    path = _cache_path(tic_id, cache_dir)
    if not path.exists():
        return None

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    row = rows[0]  # one row per star
    params = _from_row(row)

    if _is_stale(params["fetch_timestamp"]):
        return None  # treat as cache miss; fetcher will refresh

    return params


def save_to_cache(
    params: dict,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> None:
    """
    Save stellar parameters to cache as CSV.

    Args:
        params:    dict with Quantity fields (same shape as _from_row output)
        cache_dir: directory to write into (created if absent)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(params["tic_id"], cache_dir)
    row = _to_row(params)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def invalidate_cache(
    tic_id: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> bool:
    """
    Delete the cache file for tic_id.

    Returns:
        True if a file was deleted, False if nothing existed.
    """
    path = _cache_path(tic_id, cache_dir)
    if path.exists():
        path.unlink()
        return True
    return False
