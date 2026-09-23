"""
classifier.py — Zone Classifier (Stage 4)

Converts Monte Carlo physical bounds into Zone 1/2/3 boundaries.

Scalar period, duration, and depth bounds are retained for TLS compatibility.
When the sampler provides a period-duration surface, the classifier also
constructs a percentile envelope at every period-grid point.
"""

import numpy as np

from pce.schemas import (
    BoundsDistribution,
    PeriodDurationEnvelope,
    ZoneBounds,
)
from utils.constants import ZONE1_PERCENTILE, ZONE2_PERCENTILE


def classify_zones(
    bounds_dist: BoundsDistribution,
    zone1_percentile: float = ZONE1_PERCENTILE,
    zone2_percentile: float = ZONE2_PERCENTILE,
) -> tuple[ZoneBounds, ZoneBounds, ZoneBounds]:
    """
    Assign Zone 1 / Zone 2 / Zone 3 from Monte Carlo bounds.

    The scalar bounds preserve the existing API. If a period-duration surface
    is present, each zone receives a period-dependent duration envelope.

    Zone 3 remains represented by scalar outer boundaries; its physical
    envelope is intentionally omitted because the current ZoneBounds model
    represents one contiguous interval, while the complement of Zone 2 is
    generally disjoint.
    """
    if not (0 < zone2_percentile < zone1_percentile < 1):
        raise ValueError(
            f"Require 0 < zone2_percentile < zone1_percentile < 1, "
            f"got zone1={zone1_percentile}, zone2={zone2_percentile}"
        )

    z1_lo = ((1.0 - zone1_percentile) / 2.0) * 100
    z1_hi = ((1.0 + zone1_percentile) / 2.0) * 100
    z2_lo = ((1.0 - zone2_percentile) / 2.0) * 100
    z2_hi = ((1.0 + zone2_percentile) / 2.0) * 100

    def _pct(arr: np.ndarray, p: float) -> float:
        return float(np.percentile(arr, p))

    # Scalar period bounds.
    z1_period_min = _pct(bounds_dist.period_min_samples, z1_hi)
    z1_period_max = _pct(bounds_dist.period_max_samples, z1_lo)
    z2_period_min = _pct(bounds_dist.period_min_samples, z2_hi)
    z2_period_max = _pct(bounds_dist.period_max_samples, z2_lo)

    # Scalar duration bounds.
    z1_duration_min = _pct(bounds_dist.duration_min_samples, z1_hi)
    z1_duration_max = _pct(bounds_dist.duration_max_samples, z1_lo)
    z2_duration_min = _pct(bounds_dist.duration_min_samples, z2_hi)
    z2_duration_max = _pct(bounds_dist.duration_max_samples, z2_lo)

    # Scalar depth bounds.
    z1_depth_min = _pct(bounds_dist.depth_min_samples, z1_lo)
    z1_depth_max = _pct(bounds_dist.depth_max_samples, z1_hi)
    z2_depth_min = _pct(bounds_dist.depth_min_samples, z2_lo)
    z2_depth_max = _pct(bounds_dist.depth_max_samples, z2_hi)

    def _surface_envelope(
        zone_percentile: float,
        direction: str,
    ) -> PeriodDurationEnvelope | None:
        if (
            bounds_dist.duration_surface_periods is None
            or bounds_dist.duration_surface_min_hr is None
            or bounds_dist.duration_surface_max_hr is None
        ):
            return None

        periods = bounds_dist.duration_surface_periods
        surface_min = bounds_dist.duration_surface_min_hr
        surface_max = bounds_dist.duration_surface_max_hr

        if direction == "zone1":
            min_pct = ((1.0 + zone_percentile) / 2.0) * 100
            max_pct = ((1.0 - zone_percentile) / 2.0) * 100
        else:
            min_pct = ((1.0 + zone_percentile) / 2.0) * 100
            max_pct = ((1.0 - zone_percentile) / 2.0) * 100

        envelope_min = np.percentile(surface_min, min_pct, axis=0)
        envelope_max = np.percentile(surface_max, max_pct, axis=0)

        if np.any(envelope_min >= envelope_max):
            raise ValueError(
                f"{direction} period-duration envelope contains crossed "
                "duration bounds at one or more periods"
            )

        return PeriodDurationEnvelope(
            periods=periods.copy(),
            duration_min=np.asarray(envelope_min, dtype=float),
            duration_max=np.asarray(envelope_max, dtype=float),
        )

    zone1 = ZoneBounds(
        period_min=z1_period_min,
        period_max=z1_period_max,
        duration_min=z1_duration_min,
        duration_max=z1_duration_max,
        depth_min=z1_depth_min,
        depth_max=z1_depth_max,
        period_duration_envelope=_surface_envelope(zone1_percentile, "zone1"),
    )

    zone2 = ZoneBounds(
        period_min=z2_period_min,
        period_max=z2_period_max,
        duration_min=z2_duration_min,
        duration_max=z2_duration_max,
        depth_min=z2_depth_min,
        depth_max=z2_depth_max,
        period_duration_envelope=_surface_envelope(zone2_percentile, "zone2"),
    )

    zone3 = ZoneBounds(
        period_min=None,
        period_max=z2_period_min,
        duration_min=None,
        duration_max=z2_duration_min,
        depth_min=None,
        depth_max=z2_depth_min,
    )

    return zone1, zone2, zone3
