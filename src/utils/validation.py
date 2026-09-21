"""
validation.py — Shared Validation Helpers for PCE

Reusable guard functions used by fetcher.py and schemas.py.
All functions raise ValueError with a clear message on failure.
They do not return anything — call them for side effects only.
"""

from astropy import units as u
from astropy.units import Quantity


# ---------------------------------------------------------------------------
# Quantity checks
# ---------------------------------------------------------------------------

def require_quantity(value: object, name: str) -> None:
    """Raise TypeError if value is not an astropy Quantity."""
    if not isinstance(value, Quantity):
        raise TypeError(
            f"{name} must be an astropy Quantity, got {type(value).__name__}"
        )


def require_positive(value: Quantity, name: str) -> None:
    """Raise ValueError if Quantity value is not strictly positive."""
    require_quantity(value, name)
    if value.value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def require_non_negative(value: Quantity, name: str) -> None:
    """Raise ValueError if Quantity value is negative."""
    require_quantity(value, name)
    if value.value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def require_unit_physical_type(value: Quantity, name: str, physical_type: str) -> None:
    """
    Raise ValueError if Quantity does not match the expected physical type.

    Args:
        value:         The Quantity to check
        name:          Field name for error messages
        physical_type: e.g. 'mass', 'length', 'temperature', 'power'
    """
    require_quantity(value, name)
    if value.unit.physical_type != physical_type:
        raise ValueError(
            f"{name} must have physical type '{physical_type}', "
            f"got '{value.unit.physical_type}' (unit: {value.unit})"
        )


# ---------------------------------------------------------------------------
# Stellar parameter range checks
# Ranges are intentionally wide to catch catalog errors, not science errors.
# Sources: typical ranges seen in TIC-8 / confirmed exoplanet host literature
# ---------------------------------------------------------------------------

# Mass: 0.08 M_sun (hydrogen-burning limit) to 100 M_sun (massive star)
_MASS_MIN = 0.08 * u.M_sun
_MASS_MAX = 100.0 * u.M_sun

# Radius: 0.1 R_sun (late M-dwarf) to 1000 R_sun (red supergiant)
_RADIUS_MIN = 0.1 * u.R_sun
_RADIUS_MAX = 1000.0 * u.R_sun

# Teff: 2300 K (late M-dwarf) to 50 000 K (hot O-star)
_TEFF_MIN = 2300.0 * u.K
_TEFF_MAX = 50_000.0 * u.K

# Luminosity: 0.0001 L_sun (ultra-cool dwarfs, e.g. TRAPPIST-1 at 0.00052 L_sun)
# to 1e6 L_sun (luminous blue variable).
_LUMINOSITY_MIN = 0.0001 * u.L_sun
_LUMINOSITY_MAX = 1_000_000.0 * u.L_sun


def validate_stellar_mass(mass: Quantity) -> None:
    """Check mass is physically plausible for a star."""
    require_positive(mass, "mass")
    mass_solar = mass.to(u.M_sun)
    if not (_MASS_MIN <= mass_solar <= _MASS_MAX):
        raise ValueError(
            f"Stellar mass {mass_solar:.3f} outside plausible range "
            f"[{_MASS_MIN}, {_MASS_MAX}]. Check catalog data."
        )


def validate_stellar_radius(radius: Quantity) -> None:
    """Check radius is physically plausible for a star."""
    require_positive(radius, "radius")
    radius_solar = radius.to(u.R_sun)
    if not (_RADIUS_MIN <= radius_solar <= _RADIUS_MAX):
        raise ValueError(
            f"Stellar radius {radius_solar:.3f} outside plausible range "
            f"[{_RADIUS_MIN}, {_RADIUS_MAX}]. Check catalog data."
        )


def validate_stellar_teff(teff: Quantity) -> None:
    """Check effective temperature is physically plausible for a star."""
    require_positive(teff, "teff")
    teff_k = teff.to(u.K)
    if not (_TEFF_MIN <= teff_k <= _TEFF_MAX):
        raise ValueError(
            f"Stellar Teff {teff_k:.1f} outside plausible range "
            f"[{_TEFF_MIN}, {_TEFF_MAX}]. Check catalog data."
        )


def validate_stellar_luminosity(luminosity: Quantity) -> None:
    """Check luminosity is physically plausible for a star."""
    require_positive(luminosity, "luminosity")
    lum_solar = luminosity.to(u.L_sun)
    if not (_LUMINOSITY_MIN <= lum_solar <= _LUMINOSITY_MAX):
        raise ValueError(
            f"Stellar luminosity {lum_solar:.4f} outside plausible range "
            f"[{_LUMINOSITY_MIN}, {_LUMINOSITY_MAX}]. Check catalog data."
        )


def validate_uncertainty(value: Quantity, name: str) -> None:
    """
    Check that an uncertainty value is present and non-negative.
    Uncertainties of exactly 0.0 are allowed (some catalogs report this
    for well-constrained parameters) but None / NaN are not.
    """
    require_quantity(value, name)
    import math
    if math.isnan(value.value):
        raise ValueError(f"Uncertainty {name} is NaN — missing from catalog")
    require_non_negative(value, name)


def validate_all_stellar_params(
    mass: Quantity,
    radius: Quantity,
    teff: Quantity,
    luminosity: Quantity,
    mass_err_lo: Quantity,
    mass_err_hi: Quantity,
    radius_err_lo: Quantity,
    radius_err_hi: Quantity,
    teff_err_lo: Quantity,
    teff_err_hi: Quantity,
    luminosity_err_lo: Quantity,
    luminosity_err_hi: Quantity,
) -> None:
    """
    Run all stellar parameter checks in one call.
    Raises ValueError on the first failure encountered.
    Used by fetcher.py before constructing StellarParameters.
    """
    validate_stellar_mass(mass)
    validate_stellar_radius(radius)
    validate_stellar_teff(teff)
    validate_stellar_luminosity(luminosity)

    for val, name in [
        (mass_err_lo, "mass_err_lo"),
        (mass_err_hi, "mass_err_hi"),
        (radius_err_lo, "radius_err_lo"),
        (radius_err_hi, "radius_err_hi"),
        (teff_err_lo, "teff_err_lo"),
        (teff_err_hi, "teff_err_hi"),
        (luminosity_err_lo, "luminosity_err_lo"),
        (luminosity_err_hi, "luminosity_err_hi"),
    ]:
        validate_uncertainty(val, name)
