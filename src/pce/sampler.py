"""
sampler.py — Vectorised Monte Carlo Sampler (Stage 3)

Draws n_samples from the stellar parameter uncertainty distributions,
runs all 7 physics functions across every sample in one vectorised pass,
and returns a BoundsDistribution of shape (n_samples,) arrays.

Public API:
    run_monte_carlo(stellar_params, observation_baseline_days, ...) -> BoundsDistribution

Sampling strategy:
    TIC-8 uncertainties (eneg/epos) → split-normal distribution
        Below mean: N(mean, sigma_lo)
        Above mean: N(mean, sigma_hi)
    Gaia uncertainties (16th/84th percentile-derived) → same split-normal

Vectorisation strategy:
    All physics functions operate on numpy arrays directly.
    No Python loops over samples — full numpy broadcasting throughout.
    ~10-100x faster than per-sample calls.

Planet density sampling:
    Roche limit requires planet density.
    We sample log-uniformly over [PLANET_DENSITY_MIN, PLANET_DENSITY_MAX]
    to give equal weight across orders of magnitude.
"""

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from pce.schemas import StellarParameters, BoundsDistribution
from utils.constants import (
    G,
    PLANET_DENSITY_MIN,
    PLANET_DENSITY_MAX,
    MIN_TRANSITS_DEFAULT,
    MIN_TRANSIT_DEPTH,
    N_MC_SAMPLES_DEFAULT,
    MC_SEED_DEFAULT,
)
from pce.physics import R_PLANET_MAX
import math


# ---------------------------------------------------------------------------
# Split-normal sampler
# ---------------------------------------------------------------------------

def _sample_split_normal(
    mean: float,
    sigma_lo: float,
    sigma_hi: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw n samples from a split-normal distribution.

    Below mean: N(mean, sigma_lo)
    Above mean: N(mean, sigma_hi)

    This handles asymmetric uncertainties correctly without assuming symmetry.

    Args:
        mean:     Central value
        sigma_lo: Lower (negative side) standard deviation (positive float)
        sigma_hi: Upper (positive side) standard deviation (positive float)
        n:        Number of samples
        rng:      Numpy random generator

    Returns:
        np.ndarray of shape (n,)
    """
    raw = rng.standard_normal(n)
    samples = np.where(raw < 0, mean + raw * sigma_lo, mean + raw * sigma_hi)
    return samples


# ---------------------------------------------------------------------------
# Vectorised physics — array equivalents of physics.py scalar functions
# All operate on np.ndarray inputs, return np.ndarray outputs.
# Units stripped for performance — see physics.py for unit documentation.
# ---------------------------------------------------------------------------

def _vec_semi_major_axis(
    period_s: np.ndarray,
    mass_kg: np.ndarray,
    G_val: float,
) -> np.ndarray:
    """
    Vectorised Kepler 3rd law.
    Inputs in SI (seconds, kg). Output in metres.
    """
    return ((G_val * mass_kg * period_s**2) / (4 * math.pi**2)) ** (1.0 / 3.0)


def _vec_roche_period(planet_density_cgs: np.ndarray) -> np.ndarray:
    """
    Vectorised Roche limit period.
    Input: planet density in g/cm^3. Output: period in hours.
    Formula: P = 12.6 hr * (rho / 1 g/cm^3)^(-0.5)
    """
    return 12.6 * planet_density_cgs ** (-0.5)


def _vec_transit_duration_central(
    period_s: np.ndarray,
    star_radius_m: np.ndarray,
    planet_radius_m: np.ndarray,
    a_m: np.ndarray,
) -> np.ndarray:
    """
    Vectorised central transit duration (b=0).
    All inputs in SI. Output in hours.
    """
    chord = (star_radius_m + planet_radius_m) / a_m
    # Clamp chord to [0, 1) to avoid arcsin domain errors at boundaries
    chord = np.clip(chord, 0.0, 1.0 - 1e-10)
    duration_s = (period_s / math.pi) * np.arcsin(chord)
    return duration_s / 3600.0  # seconds → hours


def _vec_transit_duration_grazing(
    period_s: np.ndarray,
    star_radius_m: np.ndarray,
    planet_radius_m: np.ndarray,
    a_m: np.ndarray,
) -> np.ndarray:
    """
    Vectorised grazing onset transit duration (b = 1 - Rp/R_star).
    All inputs in SI. Output in hours.
    """
    chord = (2.0 * np.sqrt(star_radius_m * planet_radius_m)) / a_m
    chord = np.clip(chord, 0.0, 1.0 - 1e-10)
    duration_s = (period_s / math.pi) * np.arcsin(chord)
    return duration_s / 3600.0  # seconds → hours


def _vec_transit_depth(
    planet_radius_m: np.ndarray,
    star_radius_m: np.ndarray,
) -> np.ndarray:
    """
    Vectorised transit depth (uniform disk).
    Output: dimensionless.
    """
    return (planet_radius_m / star_radius_m) ** 2


def _vec_min_planet_radius(
    star_radius_m: np.ndarray,
    depth_floor: float,
) -> np.ndarray:
    """
    Vectorised minimum detectable planet radius.
    Output in metres.
    """
    return star_radius_m * math.sqrt(depth_floor)


def _vec_max_period(
    baseline_s: float,
    n_min_transits: int,
) -> float:
    """
    Maximum period from baseline — scalar (doesn't depend on stellar params).
    Output in days.
    """
    return (baseline_s / n_min_transits) / 86400.0  # seconds → days


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_monte_carlo(
    stellar_params: StellarParameters,
    observation_baseline_days: float,
    min_transits: int = MIN_TRANSITS_DEFAULT,
    min_transit_depth: float = MIN_TRANSIT_DEPTH,
    n_samples: int = N_MC_SAMPLES_DEFAULT,
    seed: int = MC_SEED_DEFAULT,
) -> BoundsDistribution:
    """
    Run vectorised Monte Carlo over stellar parameter uncertainties.

    Draws n_samples from split-normal distributions for each stellar
    parameter, then computes all physics bounds across every sample
    in a single vectorised pass.

    Args:
        stellar_params:            Validated StellarParameters from fetcher
        observation_baseline_days: Total observation window [days]
        min_transits:              Minimum transits required (2 or 3)
        min_transit_depth:         Minimum detectable depth [dimensionless]
        n_samples:                 Monte Carlo sample count (default 10 000)
        seed:                      Random seed for reproducibility

    Returns:
        BoundsDistribution — six arrays of shape (n_samples,)

    Raises:
        ValueError: if inputs are out of valid range
    """
    if observation_baseline_days <= 0:
        raise ValueError(
            f"observation_baseline_days must be > 0, got {observation_baseline_days}"
        )
    if min_transits < 2:
        raise ValueError(f"min_transits must be >= 2, got {min_transits}")
    if min_transit_depth <= 0:
        raise ValueError(f"min_transit_depth must be > 0, got {min_transit_depth}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Extract central values and uncertainties (strip units → floats)
    # ------------------------------------------------------------------
    mass_mean   = stellar_params.mass.to(u.kg).value
    mass_lo     = stellar_params.mass_err_lo.to(u.kg).value
    mass_hi     = stellar_params.mass_err_hi.to(u.kg).value

    radius_mean = stellar_params.radius.to(u.m).value
    radius_lo   = stellar_params.radius_err_lo.to(u.m).value
    radius_hi   = stellar_params.radius_err_hi.to(u.m).value

    # ------------------------------------------------------------------
    # 2. Draw stellar parameter samples — split-normal per parameter
    # ------------------------------------------------------------------
    mass_samples   = _sample_split_normal(mass_mean,   mass_lo,   mass_hi,   n_samples, rng)
    radius_samples = _sample_split_normal(radius_mean, radius_lo, radius_hi, n_samples, rng)

    # Guard: mass and radius must remain positive after sampling
    mass_samples   = np.clip(mass_samples,   mass_mean   * 0.01, None)
    radius_samples = np.clip(radius_samples, radius_mean * 0.01, None)

    # ------------------------------------------------------------------
    # 3. Planet density samples — log-uniform over plausible range
    # ------------------------------------------------------------------
    rho_min_cgs = PLANET_DENSITY_MIN.to(u.g / u.cm**3).value
    rho_max_cgs = PLANET_DENSITY_MAX.to(u.g / u.cm**3).value

    log_rho_samples = rng.uniform(
        np.log10(rho_min_cgs),
        np.log10(rho_max_cgs),
        n_samples,
    )
    planet_density_samples = 10.0 ** log_rho_samples  # g/cm^3

    # ------------------------------------------------------------------
    # 4. Planet radius samples — log-uniform over [R_p_min, R_p_max]
    #    R_p_min is star-dependent: R_star * sqrt(depth_floor)
    # ------------------------------------------------------------------
    r_p_min_samples_m = _vec_min_planet_radius(radius_samples, min_transit_depth)
    r_p_max_m = R_PLANET_MAX.to(u.m).value  # scalar

    # Use geometric mean of per-sample r_p_min for log-uniform lower bound
    # (we sample one representative planet radius per stellar sample)
    log_rp_samples = rng.uniform(
        np.log10(r_p_min_samples_m),
        np.log10(r_p_max_m),
        n_samples,
    )
    planet_radius_samples_m = 10.0 ** log_rp_samples

    # ------------------------------------------------------------------
    # 5. Constants in SI
    # ------------------------------------------------------------------
    G_val         = G.to(u.m**3 / (u.kg * u.s**2)).value
    baseline_s    = observation_baseline_days * 86400.0  # days → seconds

    # ------------------------------------------------------------------
    # 6. Compute semi-major axes across all samples
    # ------------------------------------------------------------------

    # For Roche period: use minimum Roche period (highest planet density)
    # We want the tightest (smallest) period floor, so use density that gives
    # the shortest Roche period — i.e. highest density sample
    # Both min and max bounds are stored for the classifier to use

    # Period bounds from physics
    # period_min: Roche limit — varies with planet density samples
    roche_period_hr_samples = _vec_roche_period(planet_density_samples)
    period_min_samples = roche_period_hr_samples / 24.0  # hours → days

    # period_max: baseline / min_transits — same for all samples (no stellar dependency)
    period_max_scalar = _vec_max_period(baseline_s, min_transits)
    period_max_samples = np.full(n_samples, period_max_scalar)

    # ------------------------------------------------------------------
    # 7. Transit durations — computed at representative planet radius
    #    using sampled stellar mass and radius
    # ------------------------------------------------------------------
    # Use sampled planet radii and stellar radii for duration

    # Semi-major axis at the Roche period (inner boundary)
    roche_period_s_samples = roche_period_hr_samples * 3600.0
    a_roche_m = _vec_semi_major_axis(roche_period_s_samples, mass_samples, G_val)

    # Semi-major axis at period_max (outer boundary)
    period_max_s = period_max_scalar * 86400.0
    a_max_m = _vec_semi_major_axis(
        np.full(n_samples, period_max_s), mass_samples, G_val
    )

    # duration_max: central transit at outer boundary (longest possible transit)
    duration_max_samples = _vec_transit_duration_central(
        np.full(n_samples, period_max_s),
        radius_samples,
        planet_radius_samples_m,
        a_max_m,
    )

    # duration_min: grazing onset at inner (Roche) boundary (shortest plausible)
    duration_min_samples = _vec_transit_duration_grazing(
        roche_period_s_samples,
        radius_samples,
        planet_radius_samples_m,
        a_roche_m,
    )

    # ------------------------------------------------------------------
    # 8. Depth bounds
    # ------------------------------------------------------------------
    depth_min_samples = np.full(n_samples, min_transit_depth)
    depth_max_samples = _vec_transit_depth(planet_radius_samples_m, radius_samples)

    # Guard: depth_max must be < 1 (can't block more than 100% of star)
    depth_max_samples = np.clip(depth_max_samples, 0.0, 0.999)

    # Guard: duration_min must be < duration_max per sample
    # Where they cross, clamp duration_min to 90% of duration_max
    crossed = duration_min_samples >= duration_max_samples
    duration_min_samples = np.where(
        crossed, duration_max_samples * 0.9, duration_min_samples
    )

    # ------------------------------------------------------------------
    # 9. Return BoundsDistribution
    # ------------------------------------------------------------------
    return BoundsDistribution(
        n_samples=n_samples,
        seed=seed,
        period_min_samples=period_min_samples,
        period_max_samples=period_max_samples,
        duration_min_samples=duration_min_samples,
        duration_max_samples=duration_max_samples,
        depth_min_samples=depth_min_samples,
        depth_max_samples=depth_max_samples,
    )
