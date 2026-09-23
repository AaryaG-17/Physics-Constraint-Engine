"""
physics.py — Pure Physics Functions for PCE

All functions are stateless and operate on scalar astropy Quantity inputs.
No I/O, no randomness, no side effects.

The Monte Carlo sampler (sampler.py) calls these functions across arrays
of stellar parameter samples. Functions here are the single source of truth
for every formula PCE uses.

Sources
-------
Roche limit:
    Rappaport et al. (2013), ApJ, 752, 1 — fluid-body formulation
    P_Roche = 12.6 hr * (rho_planet / 1 g/cm^3)^(-1/2)

Kepler's 3rd law:
    Standard Newtonian — P^2 = 4*pi^2*a^3 / (G*M_star)
    Planet mass dropped (M_p/M_star negligible for all planetary masses)
    Circular orbits assumed (e=0), consistent with TLS default

Transit duration:
    Winn (2010), in Exoplanets, ed. Seager — eq. for T14 (first to last contact)
    Accurate to ~27 sec (Eastman et al. 2019, EXOFASTv2)
    PCE uses the full->grazing transition boundary (b = 1 - Rp/R_star) as
    engineering bound for duration_min (grazing onset) and b=0 for
    duration_max (central transit)

Transit depth:
    Uniform-disk approximation: delta = (Rp / R_star)^2
    Limb darkening handled by TLS downstream, not PCE

Planet radius bounds:
    R_p_max = 2.2 R_Jupiter  (Hou & Wei 2022, tidal-heating inflation limit)
    R_p_min = R_star * sqrt(min_transit_depth)  (star-dependent, depth-floor driven)

Maximum period:
    P_max = observation_baseline / N_min_transits
    Engineering convention compatible with TLS (Hippke & Heller 2019, A&A 623 A39)
"""

import math

from astropy import units as u
from astropy.units import Quantity

from utils.constants import G

# ---------------------------------------------------------------------------
# Planet radius bounds (physical limits)
# ---------------------------------------------------------------------------

#: Maximum planet radius — tidal-heating inflation limit
#: Source: Hou & Wei (2022)
R_PLANET_MAX = 2.2 * u.R_jup

#: 1 g/cm^3 reference density for Roche formula normalisation
_RHO_REF = 1.0 * u.g / u.cm**3

#: 12.6 hours — Roche formula pre-factor (Rappaport et al. 2013)
_ROCHE_PREFACTOR = 12.6 * u.hour


# ---------------------------------------------------------------------------
# 1. Roche limit period
# ---------------------------------------------------------------------------

def roche_limit_period(planet_density: Quantity) -> Quantity:
    """
    Minimum orbital period set by the Roche limit (fluid-body formulation).

    Formula:
        P_Roche = 12.6 hr * (rho_planet / 1 g/cm^3)^(-1/2)

    Source: Rappaport et al. (2013), ApJ, 752, 1

    Args:
        planet_density: Planet bulk density [any density unit]

    Returns:
        Minimum orbital period [hours]

    Raises:
        ValueError: if planet_density <= 0
    """
    rho = planet_density.to(u.g / u.cm**3)
    if rho.value <= 0:
        raise ValueError(f"planet_density must be > 0, got {planet_density}")
    return _ROCHE_PREFACTOR * (rho / _RHO_REF) ** (-0.5)


# ---------------------------------------------------------------------------
# 2. Semi-major axis from Kepler's 3rd law
# ---------------------------------------------------------------------------

def semi_major_axis(period: Quantity, star_mass: Quantity) -> Quantity:
    """
    Semi-major axis from Kepler's 3rd law.

    Formula:
        a = (G * M_star * P^2 / (4 * pi^2))^(1/3)

    Assumptions:
        - Circular orbit (e = 0)
        - Planet mass negligible (M_p / M_star << 1)

    Args:
        period:    Orbital period [any time unit]
        star_mass: Stellar mass [any mass unit]

    Returns:
        Semi-major axis [metres]

    Raises:
        ValueError: if period or star_mass <= 0
    """
    P = period.to(u.s)
    M = star_mass.to(u.kg)

    if P.value <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if M.value <= 0:
        raise ValueError(f"star_mass must be > 0, got {star_mass}")

    a_cubed = (G * M * P**2) / (4 * math.pi**2)
    return a_cubed ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# 3. Transit duration — central transit (impact parameter b = 0)
# ---------------------------------------------------------------------------

def transit_duration_central(
    period: Quantity,
    star_radius: Quantity,
    planet_radius: Quantity,
    star_mass: Quantity,
) -> Quantity:
    """
    Duration of a central transit (b = 0, first-to-last contact, T14).

    Formula (Winn 2010):
        T14 = (P / pi) * arcsin[(R_star / a) * sqrt((1 + Rp/R_star)^2)]
            = (P / pi) * arcsin[(R_star + Rp) / a]

    This is the maximum transit duration for a given period — planet crosses
    the full stellar diameter.

    Source: Winn (2010), in Exoplanets (ed. Seager), eq. T14 with b=0

    Args:
        period:        Orbital period [any time unit]
        star_radius:   Stellar radius [any length unit]
        planet_radius: Planet radius [any length unit]
        star_mass:     Stellar mass [any mass unit]

    Returns:
        Transit duration [hours]

    Raises:
        ValueError: if any input <= 0, or if (R_star + Rp) > a (unphysical)
    """
    if star_radius.to(u.m).value <= 0:
        raise ValueError(f"star_radius must be > 0, got {star_radius}")
    if planet_radius.to(u.m).value <= 0:
        raise ValueError(f"planet_radius must be > 0, got {planet_radius}")

    a = semi_major_axis(period, star_mass)

    R_s = star_radius.to(u.m)
    R_p = planet_radius.to(u.m)
    P = period.to(u.s)

    chord = (R_s + R_p) / a  # dimensionless

    if chord.value >= 1.0:
        raise ValueError(
            f"(R_star + R_planet) / a = {chord.value:.4f} >= 1 — "
            f"planet orbit is inside the star. "
            f"R_star={star_radius}, R_planet={planet_radius}, a={a.to(u.au):.4f}"
        )

    duration = (P / math.pi) * math.asin(chord.value)
    return duration.to(u.hour)


# ---------------------------------------------------------------------------
# 4. Transit duration — grazing onset (impact parameter b = 1 - Rp/R_star)
# ---------------------------------------------------------------------------

def transit_duration_grazing_onset(
    period: Quantity,
    star_radius: Quantity,
    planet_radius: Quantity,
    star_mass: Quantity,
) -> Quantity:
    """
    Duration at the grazing onset boundary (b = 1 - Rp/R_star, T14).

    At this impact parameter the planet's limb just touches the stellar limb
    at mid-transit — the boundary between full and grazing transits.
    PCE uses this as the minimum plausible transit duration.

    Formula (Winn 2010, T14 with b = 1 - Rp/R_star):
        chord = (R_star / a) * sqrt((1 + Rp/R_star)^2 - (1 - Rp/R_star)^2)
              = (R_star / a) * sqrt(4 * Rp/R_star)
              = (2 / a) * sqrt(R_star * Rp)
        T14   = (P / pi) * arcsin(chord)

    Source: Winn (2010) — PCE engineering interpretation of the b boundary.
    Verified: Eastman et al. (2019) EXOFASTv2, accuracy ~27 sec.

    Args:
        period:        Orbital period [any time unit]
        star_radius:   Stellar radius [any length unit]
        planet_radius: Planet radius [any length unit]
        star_mass:     Stellar mass [any mass unit]

    Returns:
        Minimum transit duration at grazing onset [hours]

    Raises:
        ValueError: if any input <= 0, or chord >= 1
    """
    if star_radius.to(u.m).value <= 0:
        raise ValueError(f"star_radius must be > 0, got {star_radius}")
    if planet_radius.to(u.m).value <= 0:
        raise ValueError(f"planet_radius must be > 0, got {planet_radius}")

    a = semi_major_axis(period, star_mass)

    R_s = star_radius.to(u.m)
    R_p = planet_radius.to(u.m)
    P = period.to(u.s)

    # chord = 2 * sqrt(R_star * R_p) / a
    chord = (2.0 * (R_s * R_p) ** 0.5) / a  # dimensionless

    if chord.value >= 1.0:
        raise ValueError(
            f"Grazing chord (2*sqrt(R_star*R_planet)/a) = {chord.value:.4f} >= 1. "
            f"R_star={star_radius}, R_planet={planet_radius}, a={a.to(u.au):.4f}"
        )

    duration = (P / math.pi) * math.asin(chord.value)
    return duration.to(u.hour)


# ---------------------------------------------------------------------------
# 5. Transit depth
# ---------------------------------------------------------------------------

def transit_depth(planet_radius: Quantity, star_radius: Quantity) -> float:
    """
    Fractional flux drop during transit — uniform-disk approximation.

    Formula:
        delta = (Rp / R_star)^2

    Limb darkening is handled by TLS downstream.

    Args:
        planet_radius: Planet radius [any length unit]
        star_radius:   Stellar radius [any length unit — same unit as planet_radius]

    Returns:
        Transit depth [dimensionless, 0 < delta < 1]

    Raises:
        ValueError: if either radius <= 0, or Rp > R_star (depth would be > 1)
    """
    R_p = planet_radius.to(u.m).value
    R_s = star_radius.to(u.m).value

    if R_p <= 0:
        raise ValueError(f"planet_radius must be > 0, got {planet_radius}")
    if R_s <= 0:
        raise ValueError(f"star_radius must be > 0, got {star_radius}")
    if R_p > R_s:
        raise ValueError(
            f"planet_radius ({planet_radius}) > star_radius ({star_radius}) — "
            f"depth would exceed 1.0. Unphysical."
        )

    return (R_p / R_s) ** 2


# ---------------------------------------------------------------------------
# 6. Minimum detectable planet radius
# ---------------------------------------------------------------------------

def minimum_detectable_planet_radius(
    star_radius: Quantity,
    depth_floor: float,
) -> Quantity:
    """
    Smallest planet radius detectable given the photometric depth floor.

    Formula:
        R_p_min = R_star * sqrt(depth_floor)

    Args:
        star_radius: Stellar radius [any length unit]
        depth_floor: Minimum detectable fractional depth [dimensionless]
                     e.g. 200e-6 for 200 ppm

    Returns:
        Minimum detectable planet radius [same unit as star_radius]

    Raises:
        ValueError: if star_radius <= 0 or depth_floor <= 0
    """
    if star_radius.to(u.m).value <= 0:
        raise ValueError(f"star_radius must be > 0, got {star_radius}")
    if depth_floor <= 0:
        raise ValueError(f"depth_floor must be > 0, got {depth_floor}")

    return star_radius * math.sqrt(depth_floor)


# ---------------------------------------------------------------------------
# 7. Maximum period from observation baseline
# ---------------------------------------------------------------------------

def maximum_period_from_baseline(
    baseline: Quantity,
    n_min_transits: int,
) -> Quantity:
    """
    Maximum detectable orbital period given observation baseline and minimum
    required transit count.

    Formula:
        P_max = baseline / n_min_transits

    Engineering convention compatible with TLS (Hippke & Heller 2019).
    Note: first-transit phase can cause edge exceptions; this is the standard
    TLS-compatible approximation.

    Args:
        baseline:       Total observation window [any time unit]
        n_min_transits: Minimum number of transits required (2 or 3)

    Returns:
        Maximum orbital period [days]

    Raises:
        ValueError: if baseline <= 0 or n_min_transits < 2
    """
    if baseline.to(u.day).value <= 0:
        raise ValueError(f"baseline must be > 0, got {baseline}")
    if n_min_transits < 2:
        raise ValueError(
            f"n_min_transits must be >= 2 (TLS requires at least 2 transits), "
            f"got {n_min_transits}"
        )

    return (baseline / n_min_transits).to(u.day)
