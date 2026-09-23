"""
sampler.py — Vectorised Monte Carlo Sampler

Monte Carlo propagates uncertainty in the stellar parameters only.

Planetary assumptions used to construct physical bounds are deterministic:

    - PLANET_DENSITY_MAX -> minimum Roche period
    - min_transit_depth  -> minimum detectable planet radius
    - R_PLANET_MAX       -> maximum planet radius

This is intentional. Randomly sampling planet properties and taking the
resulting extrema would make the physical search boundary depend on random
draws rather than on the adopted physical prior.

Public API:
    run_monte_carlo(stellar_params, observation_baseline_days, ...)
        -> BoundsDistribution
"""

import numpy as np
from astropy import units as u

from pce.physics import (
    R_PLANET_MAX,
    _vectorized_semi_major_axis,
    _vectorized_minimum_detectable_planet_radius,
    _vectorized_transit_depth,
    _vectorized_transit_duration_central,
    _vectorized_transit_duration_grazing_onset,
    maximum_period_from_baseline,
    roche_limit_period,
)
from pce.schemas import BoundsDistribution, StellarParameters
from utils.constants import (
    MIN_TRANSITS_DEFAULT,
    MIN_TRANSIT_DEPTH,
    N_MC_SAMPLES_DEFAULT,
    MC_SEED_DEFAULT,
    PLANET_DENSITY_MAX,
)


# Number of period samples used to represent the physical period-duration surface.
DURATION_SURFACE_PERIODS_DEFAULT = 128



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

    Below mean:
        N(mean, sigma_lo)

    Above mean:
        N(mean, sigma_hi)
    """
    raw = rng.standard_normal(n)

    return np.where(
        raw < 0,
        mean + raw * sigma_lo,
        mean + raw * sigma_hi,
    )


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

# Physical equations are implemented centrally in pce.physics. The sampler
# only prepares NumPy arrays, calls those equations, and validates the results.

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_valid_geometry(
    name: str,
    values: np.ndarray,
) -> None:
    """
    Fail loudly if any physical geometry calculation is invalid.
    """
    invalid = ~np.isfinite(values)

    if np.any(invalid):
        count = int(np.count_nonzero(invalid))
        total = values.size

        raise ValueError(
            f"{name} produced {count}/{total} invalid geometries. "
            "The current physical model cannot safely construct duration "
            "bounds for these stellar realizations."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_monte_carlo(
    stellar_params: StellarParameters,
    observation_baseline_days: float,
    min_transits: int = MIN_TRANSITS_DEFAULT,
    min_transit_depth: float = MIN_TRANSIT_DEPTH,
    n_samples: int = N_MC_SAMPLES_DEFAULT,
    seed: int = MC_SEED_DEFAULT,
    n_duration_periods: int = DURATION_SURFACE_PERIODS_DEFAULT,
) -> BoundsDistribution:
    """
    Propagate stellar-parameter uncertainty through PCE.

    Monte Carlo sampling is applied only to uncertain stellar parameters.
    Planet density and planet radius are deterministic physical/search
    assumptions.

    Args:
        stellar_params:
            Validated stellar parameters.

        observation_baseline_days:
            Total observation baseline [days].

        min_transits:
            Minimum number of observed transits required.

        min_transit_depth:
            Adopted minimum detectable transit depth.

        n_samples:
            Number of Monte Carlo stellar realizations.

        seed:
            Random seed for reproducibility.

        n_duration_periods:
            Number of period grid points used to construct the
            period-duration physical constraint surface.

    Returns:
        BoundsDistribution containing one set of physical bounds per
        stellar realization.

    Raises:
        ValueError:
            If inputs are invalid or any stellar realization produces
            invalid transit geometry.
    """
    if observation_baseline_days <= 0:
        raise ValueError(
            "observation_baseline_days must be > 0, "
            f"got {observation_baseline_days}"
        )

    if min_transits < 2:
        raise ValueError(
            f"min_transits must be >= 2, got {min_transits}"
        )

    if min_transit_depth <= 0:
        raise ValueError(
            f"min_transit_depth must be > 0, "
            f"got {min_transit_depth}"
        )

    if min_transit_depth >= 1:
        raise ValueError(
            f"min_transit_depth must be < 1, "
            f"got {min_transit_depth}"
        )

    if n_samples < 1:
        raise ValueError(
            f"n_samples must be >= 1, got {n_samples}"
        )

    if n_duration_periods < 2:
        raise ValueError(
            f"n_duration_periods must be >= 2, got {n_duration_periods}"
        )

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Stellar parameter distributions
    # ------------------------------------------------------------------

    mass_mean = stellar_params.mass.to(u.kg).value
    mass_lo = stellar_params.mass_err_lo.to(u.kg).value
    mass_hi = stellar_params.mass_err_hi.to(u.kg).value

    radius_mean = stellar_params.radius.to(u.m).value
    radius_lo = stellar_params.radius_err_lo.to(u.m).value
    radius_hi = stellar_params.radius_err_hi.to(u.m).value

    mass_samples = _sample_split_normal(
        mass_mean,
        mass_lo,
        mass_hi,
        n_samples,
        rng,
    )

    radius_samples = _sample_split_normal(
        radius_mean,
        radius_lo,
        radius_hi,
        n_samples,
        rng,
    )

    # Keep sampled stellar parameters physically positive.
    mass_samples = np.clip(
        mass_samples,
        mass_mean * 0.01,
        None,
    )

    radius_samples = np.clip(
        radius_samples,
        radius_mean * 0.01,
        None,
    )

    # ------------------------------------------------------------------
    # 2. Deterministic planetary assumptions
    # ------------------------------------------------------------------

    # The shortest Roche period occurs at the maximum adopted planet
    # density because P_Roche ∝ rho_planet^(-1/2).
    roche_period = roche_limit_period(
        PLANET_DENSITY_MAX
    ).to(u.day).value

    period_min_samples = np.full(
        n_samples,
        roche_period,
        dtype=np.float64,
    )

    # Maximum period from the observing baseline.
    period_max_scalar = maximum_period_from_baseline(
        observation_baseline_days * u.day,
        min_transits,
    ).value

    period_max_samples = np.full(
        n_samples,
        period_max_scalar,
        dtype=np.float64,
    )

    # Minimum detectable planet radius is determined by the adopted
    # detection floor.
    planet_radius_min_samples_m = _vectorized_minimum_detectable_planet_radius(
        radius_samples,
        min_transit_depth,
    )

    # Maximum planet radius is a deterministic project assumption.
    planet_radius_max_m = R_PLANET_MAX.to(u.m).value

    # ------------------------------------------------------------------
    # 3. Transit geometry
    # ------------------------------------------------------------------

    roche_period_s = roche_period * 86400.0
    period_max_s = period_max_scalar * 86400.0

    # ------------------------------------------------------------------
    # 4. Duration bounds
    # ------------------------------------------------------------------

    # Minimum duration:
    #   earliest allowed orbit + grazing onset + smallest detectable planet
    duration_min_samples = _vectorized_transit_duration_grazing_onset(
        np.full(
            n_samples,
            roche_period_s,
            dtype=np.float64,
        ),
        radius_samples,
        planet_radius_min_samples_m,
        mass_samples,
    )

    # Maximum duration:
    #   latest observable orbit + central transit + largest adopted planet
    duration_max_samples = _vectorized_transit_duration_central(
        np.full(
            n_samples,
            period_max_s,
            dtype=np.float64,
        ),
        radius_samples,
        np.full(
            n_samples,
            planet_radius_max_m,
            dtype=np.float64,
        ),
        mass_samples,
    )

    _require_valid_geometry(
        "duration_min",
        duration_min_samples,
    )

    _require_valid_geometry(
        "duration_max",
        duration_max_samples,
    )

    # ------------------------------------------------------------------
    # 5. Period-duration physical constraint surface
    # ------------------------------------------------------------------
    # The scalar duration bounds above are retained for backward compatibility.
    # The surface captures the actual dependence of transit duration on period.
    period_grid_days = np.geomspace(
        roche_period,
        period_max_scalar,
        num=n_duration_periods,
    )

    period_grid_s = period_grid_days * 86400.0
    period_grid_s_2d = np.broadcast_to(
        period_grid_s,
        (n_samples, n_duration_periods),
    )
    mass_2d = mass_samples[:, None]
    radius_2d = radius_samples[:, None]
    planet_radius_min_2d = planet_radius_min_samples_m[:, None]
    planet_radius_max_2d = np.full(
        (n_samples, n_duration_periods),
        planet_radius_max_m,
        dtype=np.float64,
    )

    a_surface_m = _vectorized_semi_major_axis(
        period_grid_s_2d,
        mass_2d,
    )

    duration_surface_min_hr = _vectorized_transit_duration_grazing_onset(
        period_grid_s_2d,
        radius_2d,
        planet_radius_min_2d,
        mass_2d,
    )

    duration_surface_max_hr = _vectorized_transit_duration_central(
        period_grid_s_2d,
        radius_2d,
        planet_radius_max_2d,
        mass_2d,
    )

    _require_valid_geometry(
        "duration_surface_min",
        duration_surface_min_hr,
    )
    _require_valid_geometry(
        "duration_surface_max",
        duration_surface_max_hr,
    )

    if np.any(duration_surface_min_hr >= duration_surface_max_hr):
        raise ValueError(
            "Period-duration surface contains crossed duration bounds "
            "for one or more stellar realizations."
        )

    # ------------------------------------------------------------------
    # 6. Depth bounds
    # ------------------------------------------------------------------

    # Minimum depth is the explicit detection floor.
    depth_min_samples = np.full(
        n_samples,
        min_transit_depth,
        dtype=np.float64,
    )

    # Maximum depth comes from the maximum adopted planet radius.
    depth_max_samples = _vectorized_transit_depth(
        np.full(
            n_samples,
            planet_radius_max_m,
            dtype=np.float64,
        ),
        radius_samples,
    )

    if np.any(depth_max_samples >= 1.0):
        raise ValueError(
            "Maximum adopted planet radius produces a transit depth "
            ">= 100% for one or more stellar realizations."
        )

    # ------------------------------------------------------------------
    # 7. Physical ordering checks
    # ------------------------------------------------------------------

    if np.any(period_min_samples >= period_max_samples):
        raise ValueError(
            "Roche minimum period is not below the observation-derived "
            "maximum period for all stellar realizations."
        )

    if np.any(duration_min_samples >= duration_max_samples):
        invalid_count = int(
            np.count_nonzero(
                duration_min_samples >= duration_max_samples
            )
        )

        raise ValueError(
            "Duration bounds cross for "
            f"{invalid_count}/{n_samples} stellar realizations. "
            "The model must be corrected rather than repairing the "
            "bounds numerically."
        )

    if np.any(depth_min_samples >= depth_max_samples):
        invalid_count = int(
            np.count_nonzero(
                depth_min_samples >= depth_max_samples
            )
        )

        raise ValueError(
            "Minimum transit depth is not below the maximum physical "
            f"depth for {invalid_count}/{n_samples} stellar realizations."
        )

    # ------------------------------------------------------------------
    # 8. Return
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
        duration_surface_periods=period_grid_days,
        duration_surface_min_hr=duration_surface_min_hr,
        duration_surface_max_hr=duration_surface_max_hr,
    )