"""
schemas.py — PCE Data Contracts

All Pydantic models used across the PCE pipeline.
Every stage accepts and returns one of these models.
This is the single source of truth for data shapes.

Stage flow:
    StellarParameters (fetcher.py)
        → PhysicalBounds (physics.py, per-sample scalar output)
        → BoundsDistribution (sampler.py, Monte Carlo arrays)
        → ZoneBounds (classifier.py, percentile zones)
        → ZoneMap (config.py, final TLS-ready output)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Pydantic v2 compatibility: Quantity is not a standard Python type.
# We store Quantity fields as annotated types and bypass Pydantic's default
# validator by using model_config = {"arbitrary_types_allowed": True}.
# ---------------------------------------------------------------------------

class _QuantityModel(BaseModel):
    """Base model that permits astropy Quantity and numpy ndarray fields."""

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Stage 2 output — Fetcher
# ---------------------------------------------------------------------------

class StellarParameters(_QuantityModel):
    """
    Validated stellar parameters for one star, sourced from TIC-8 or Gaia.

    All physical quantities carry astropy units.
    Asymmetric uncertainties are stored separately (lower, upper) so the
    Monte Carlo sampler can use a split-normal distribution.

    Fields ending in _err_lo / _err_hi follow TIC-8 convention:
        _err_lo: magnitude of the negative (lower) uncertainty
        _err_hi: magnitude of the positive (upper) uncertainty
    Both are positive floats; the sampler applies the sign internally.
    """

    tic_id: str = Field(..., description="TIC identifier, e.g. 'TIC_25155310'")

    # Core physical parameters
    mass: Quantity = Field(..., description="Stellar mass [M_sun]")
    mass_err_lo: Quantity = Field(..., description="Lower mass uncertainty (positive) [M_sun]")
    mass_err_hi: Quantity = Field(..., description="Upper mass uncertainty (positive) [M_sun]")

    radius: Quantity = Field(..., description="Stellar radius [R_sun]")
    radius_err_lo: Quantity = Field(..., description="Lower radius uncertainty (positive) [R_sun]")
    radius_err_hi: Quantity = Field(..., description="Upper radius uncertainty (positive) [R_sun]")

    teff: Quantity = Field(..., description="Effective temperature [K]")
    teff_err_lo: Quantity = Field(..., description="Lower Teff uncertainty (positive) [K]")
    teff_err_hi: Quantity = Field(..., description="Upper Teff uncertainty (positive) [K]")

    luminosity: Quantity = Field(..., description="Stellar luminosity [L_sun]")
    luminosity_err_lo: Quantity = Field(..., description="Lower luminosity uncertainty (positive) [L_sun]")
    luminosity_err_hi: Quantity = Field(..., description="Upper luminosity uncertainty (positive) [L_sun]")

    # Catalog metadata
    catalog_source: str = Field(
        default="TIC-8",
        description="Source catalog: 'TIC-8' or 'Gaia' or 'TIC-8+Gaia'",
    )
    catalog_version: Optional[str] = Field(
        default=None,
        description="Catalog version string if available",
    )
    fetch_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when parameters were fetched",
    )

    @field_validator("mass", "radius", "teff", "luminosity", mode="before")
    @classmethod
    def _must_be_quantity(cls, v: object) -> Quantity:
        if not isinstance(v, Quantity):
            raise TypeError(f"Expected astropy Quantity, got {type(v)}")
        return v

    @field_validator(
        "mass_err_lo", "mass_err_hi",
        "radius_err_lo", "radius_err_hi",
        "teff_err_lo", "teff_err_hi",
        "luminosity_err_lo", "luminosity_err_hi",
        mode="before",
    )
    @classmethod
    def _uncertainty_must_be_positive_quantity(cls, v: object) -> Quantity:
        if not isinstance(v, Quantity):
            raise TypeError(f"Expected astropy Quantity, got {type(v)}")
        if v.value < 0:
            raise ValueError(f"Uncertainty must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _physical_sanity(self) -> "StellarParameters":
        if self.mass.to(u.M_sun).value <= 0:
            raise ValueError(f"Stellar mass must be positive, got {self.mass}")
        if self.radius.to(u.R_sun).value <= 0:
            raise ValueError(f"Stellar radius must be positive, got {self.radius}")
        if self.teff.to(u.K).value <= 0:
            raise ValueError(f"Teff must be positive, got {self.teff}")
        if self.luminosity.to(u.L_sun).value <= 0:
            raise ValueError(f"Luminosity must be positive, got {self.luminosity}")
        return self


# ---------------------------------------------------------------------------
# Stage 1 output — Physics (scalar, point estimate for one sample)
# ---------------------------------------------------------------------------

class PhysicalBounds(BaseModel):
    """
    Physics-derived parameter bounds for one star, from a single set of
    scalar stellar parameters.

    All values are plain floats in standard units (days, hours, dimensionless).
    Quantity units are stripped at this boundary for performance — the
    sampler calls physics in tight loops over 10k samples.

    Units:
        period_*   — days
        duration_* — hours
        depth_*    — dimensionless (fractional flux drop, e.g. 200e-6 = 200 ppm)
    """

    period_min_day: float = Field(..., gt=0, description="Minimum orbital period [days] — Roche limit")
    period_max_day: float = Field(..., gt=0, description="Maximum orbital period [days] — baseline / min_transits")
    duration_min_hr: float = Field(..., gt=0, description="Minimum transit duration [hours] — grazing geometry")
    duration_max_hr: float = Field(..., gt=0, description="Maximum transit duration [hours] — central transit")
    depth_min: float = Field(..., gt=0, description="Minimum detectable depth [dimensionless]")
    depth_max: float = Field(..., gt=0, lt=1, description="Maximum physical depth [dimensionless]")

    @model_validator(mode="after")
    def _ordering(self) -> "PhysicalBounds":
        if self.period_min_day >= self.period_max_day:
            raise ValueError(
                f"period_min ({self.period_min_day:.4f} d) must be < "
                f"period_max ({self.period_max_day:.4f} d)"
            )
        if self.duration_min_hr >= self.duration_max_hr:
            raise ValueError(
                f"duration_min ({self.duration_min_hr:.4f} hr) must be < "
                f"duration_max ({self.duration_max_hr:.4f} hr)"
            )
        if self.depth_min >= self.depth_max:
            raise ValueError(
                f"depth_min ({self.depth_min:.2e}) must be < "
                f"depth_max ({self.depth_max:.2e})"
            )
        return self


# ---------------------------------------------------------------------------
# Stage 3 output — Monte Carlo Sampler
# ---------------------------------------------------------------------------

class BoundsDistribution(_QuantityModel):
    """
    Full Monte Carlo distribution of physical bounds.

    Each array has shape (n_samples,) — one value per MC sample.
    Arrays are plain numpy float64 (no units); see PhysicalBounds for units.

    Units (same as PhysicalBounds):
        period_*   — days
        duration_* — hours
        depth_*    — dimensionless
    """

    n_samples: int = Field(..., gt=0, description="Number of Monte Carlo samples drawn")
    seed: int = Field(..., description="Random seed used for reproducibility")

    # Per-sample arrays — shape (n_samples,)
    period_min_samples: np.ndarray = Field(..., description="Distribution of period_min [days]")
    period_max_samples: np.ndarray = Field(..., description="Distribution of period_max [days]")
    duration_min_samples: np.ndarray = Field(..., description="Distribution of duration_min [hours]")
    duration_max_samples: np.ndarray = Field(..., description="Distribution of duration_max [hours]")
    depth_min_samples: np.ndarray = Field(..., description="Distribution of depth_min [dimensionless]")
    depth_max_samples: np.ndarray = Field(..., description="Distribution of depth_max [dimensionless]")

    # Optional period-duration surface generated by the sampler.
    # Shape: (n_samples, n_periods).
    duration_surface_periods: Optional[np.ndarray] = Field(
        default=None,
        description="Shared period grid for the duration surface [days]",
    )
    duration_surface_min_hr: Optional[np.ndarray] = Field(
        default=None,
        description="Per-sample minimum duration across the period grid [hours]",
    )
    duration_surface_max_hr: Optional[np.ndarray] = Field(
        default=None,
        description="Per-sample maximum duration across the period grid [hours]",
    )

    @model_validator(mode="after")
    def _array_shapes_consistent(self) -> "BoundsDistribution":
        arrays = {
            "period_min_samples": self.period_min_samples,
            "period_max_samples": self.period_max_samples,
            "duration_min_samples": self.duration_min_samples,
            "duration_max_samples": self.duration_max_samples,
            "depth_min_samples": self.depth_min_samples,
            "depth_max_samples": self.depth_max_samples,
        }
        for name, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be np.ndarray, got {type(arr)}")
            if arr.shape != (self.n_samples,):
                raise ValueError(
                    f"{name} shape {arr.shape} != expected ({self.n_samples},)"
                )

        surface_fields = (
            self.duration_surface_periods,
            self.duration_surface_min_hr,
            self.duration_surface_max_hr,
        )
        if any(value is not None for value in surface_fields):
            if not all(value is not None for value in surface_fields):
                raise ValueError(
                    "duration surface fields must be provided together"
                )

            periods = self.duration_surface_periods
            surface_min = self.duration_surface_min_hr
            surface_max = self.duration_surface_max_hr

            assert periods is not None
            assert surface_min is not None
            assert surface_max is not None

            if periods.ndim != 1:
                raise ValueError("duration_surface_periods must be 1-D")

            expected_shape = (self.n_samples, periods.size)
            if surface_min.shape != expected_shape:
                raise ValueError(
                    "duration_surface_min_hr shape "
                    f"{surface_min.shape} != expected {expected_shape}"
                )
            if surface_max.shape != expected_shape:
                raise ValueError(
                    "duration_surface_max_hr shape "
                    f"{surface_max.shape} != expected {expected_shape}"
                )

        return self


# ---------------------------------------------------------------------------
# Stage 4 output — Classifier
# ---------------------------------------------------------------------------

class PeriodDurationEnvelope(_QuantityModel):
    """
    Period-dependent physical duration envelope.

    ``periods`` has shape (n_periods,). The duration arrays have shape
    (n_periods,) and define the allowed duration interval at each period.
    """

    periods: np.ndarray = Field(
        ..., description="Orbital-period grid [days]"
    )
    duration_min: np.ndarray = Field(
        ..., description="Minimum physical duration at each period [hours]"
    )
    duration_max: np.ndarray = Field(
        ..., description="Maximum physical duration at each period [hours]"
    )

    @model_validator(mode="after")
    def _validate_shapes(self) -> "PeriodDurationEnvelope":
        if not isinstance(self.periods, np.ndarray):
            raise TypeError("periods must be np.ndarray")
        if not isinstance(self.duration_min, np.ndarray):
            raise TypeError("duration_min must be np.ndarray")
        if not isinstance(self.duration_max, np.ndarray):
            raise TypeError("duration_max must be np.ndarray")

        if self.periods.ndim != 1:
            raise ValueError("periods must be one-dimensional")

        expected = self.periods.shape
        if self.duration_min.shape != expected:
            raise ValueError(
                f"duration_min shape {self.duration_min.shape} != {expected}"
            )
        if self.duration_max.shape != expected:
            raise ValueError(
                f"duration_max shape {self.duration_max.shape} != {expected}"
            )

        if len(self.periods) < 2:
            raise ValueError("period-duration envelope requires at least 2 periods")

        if not np.all(np.isfinite(self.periods)):
            raise ValueError("periods must contain only finite values")
        if not np.all(np.diff(self.periods) > 0):
            raise ValueError("periods must be strictly increasing")
        if not np.all(np.isfinite(self.duration_min)):
            raise ValueError("duration_min must contain only finite values")
        if not np.all(np.isfinite(self.duration_max)):
            raise ValueError("duration_max must contain only finite values")
        if np.any(self.duration_min >= self.duration_max):
            raise ValueError("duration_min must be < duration_max at every period")

        return self


class ZoneBounds(BaseModel):
    """
    Bounds for one zone (Zone 1, Zone 2, or Zone 3).

    Units same as PhysicalBounds (days / hours / dimensionless).
    A None value for a bound means the zone is open-ended in that direction.
    """

    period_min: Optional[float] = Field(default=None, description="Zone period lower bound [days]")
    period_max: Optional[float] = Field(default=None, description="Zone period upper bound [days]")
    duration_min: Optional[float] = Field(default=None, description="Zone duration lower bound [hours]")
    duration_max: Optional[float] = Field(default=None, description="Zone duration upper bound [hours]")
    depth_min: Optional[float] = Field(default=None, description="Zone depth lower bound [dimensionless]")
    depth_max: Optional[float] = Field(default=None, description="Zone depth upper bound [dimensionless]")
    period_duration_envelope: Optional[PeriodDurationEnvelope] = Field(
        default=None,
        description="Period-dependent physically allowed duration envelope",
    )


# ---------------------------------------------------------------------------
# Stage 5 output — TLS Config (final pipeline output)
# ---------------------------------------------------------------------------

class ZoneMap(BaseModel):
    """
    Final PCE output — TLS-ready zone configuration for one star.

    Usage:
        tls.period_min = zone_map.zone1.period_min
        tls.period_max = zone_map.zone1.period_max

    Zone semantics:
        zone1 — high confidence (≥90% of MC samples agree). Search here first.
        zone2 — uncertain edges (50–90% agreement). Search at lower resolution.
        zone3 — excluded (<50% agreement or physically implausible). Skip.

    Metadata fields document the run configuration for reproducibility.
    """

    star_id: str = Field(..., description="TIC ID or other star identifier")

    zone1: ZoneBounds = Field(..., description="High-confidence zone (≥90% MC agreement)")
    zone2: ZoneBounds = Field(..., description="Uncertain zone (50–90% MC agreement)")
    zone3: ZoneBounds = Field(..., description="Excluded zone (<50% MC agreement)")

    # Run metadata
    n_samples: int = Field(..., gt=0, description="Number of MC samples used")
    seed: int = Field(..., description="Random seed for reproducibility")
    observation_baseline_days: float = Field(..., gt=0, description="Observation baseline [days]")
    min_transits: int = Field(..., ge=2, description="Minimum transits required for detection")
    min_transit_depth: float = Field(..., gt=0, description="Minimum detectable depth [dimensionless]")
    catalog_source: str = Field(..., description="Stellar parameter source: 'TIC-8', 'Gaia', or 'TIC-8+Gaia'")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this ZoneMap was generated",
    )
