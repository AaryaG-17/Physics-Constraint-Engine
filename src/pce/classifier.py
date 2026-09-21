"""
classifier.py — Zone Classifier (Stage 4)

Converts a BoundsDistribution (Monte Carlo output) into Zone 1/2/3
boundaries using percentile thresholds.

Zone semantics:
    Zone 1 — high confidence  : ≥90% of MC samples agree
    Zone 2 — uncertain edges  : 50–90% of MC samples agree
    Zone 3 — excluded         : <50% agreement or physically implausible

Public API:
    classify_zones(bounds_dist, zone1_percentile, zone2_percentile) -> ZoneBounds tuple

For each parameter (period, duration, depth):
    Zone 1: [P10, P90]   — inner 80% of samples
    Zone 2: [P5,  P95]   — inner 90% of samples (wider than Zone 1)
    Zone 3: everything outside Zone 2

The classifier does not assign Zone 3 a hard numeric bound —
it is implicitly "everything outside Zone 2".
"""

import numpy as np
from pce.schemas import BoundsDistribution, ZoneBounds
from utils.constants import ZONE1_PERCENTILE, ZONE2_PERCENTILE


def classify_zones(
    bounds_dist: BoundsDistribution,
    zone1_percentile: float = ZONE1_PERCENTILE,
    zone2_percentile: float = ZONE2_PERCENTILE,
) -> tuple[ZoneBounds, ZoneBounds, ZoneBounds]:
    """
    Assign Zone 1 / 2 / 3 from a BoundsDistribution via percentile thresholds.

    Args:
        bounds_dist:       Output of run_monte_carlo
        zone1_percentile:  Confidence threshold for Zone 1 (default 0.90)
        zone2_percentile:  Confidence threshold for Zone 2 (default 0.50)

    Returns:
        Tuple of (zone1, zone2, zone3) ZoneBounds objects.
        zone3 bounds are the inverse of zone2 (open-ended extremes).

    Raises:
        ValueError: if percentile thresholds are out of range or inverted
    """
    if not (0 < zone2_percentile < zone1_percentile < 1):
        raise ValueError(
            f"Require 0 < zone2_percentile < zone1_percentile < 1, "
            f"got zone1={zone1_percentile}, zone2={zone2_percentile}"
        )

    # ------------------------------------------------------------------
    # Percentile levels for each zone boundary
    # Zone 1: inner (1 - zone1_percentile)/2 to (1 + zone1_percentile)/2
    #         e.g. zone1=0.90 → [P5, P95] of the min/max arrays
    # Zone 2: inner (1 - zone2_percentile)/2 to (1 + zone2_percentile)/2
    #         e.g. zone2=0.50 → [P25, P75]
    #
    # But note: we have separate arrays for _min and _max bounds.
    # For period_min we take the low percentile (tight lower bound).
    # For period_max we take the high percentile (generous upper bound).
    # ------------------------------------------------------------------

    z1_lo = ((1.0 - zone1_percentile) / 2.0) * 100  # e.g. 5.0
    z1_hi = ((1.0 + zone1_percentile) / 2.0) * 100  # e.g. 95.0
    z2_lo = ((1.0 - zone2_percentile) / 2.0) * 100  # e.g. 25.0
    z2_hi = ((1.0 + zone2_percentile) / 2.0) * 100  # e.g. 75.0

    def _pct(arr: np.ndarray, p: float) -> float:
        return float(np.percentile(arr, p))

    # ------------------------------------------------------------------
    # Period bounds
    # period_min array → lower bound of Zone 1/2 (use high percentile
    #   so Zone 1 is conservative: 90% of samples agree the min is ≤ this)
    # period_max array → upper bound of Zone 1/2 (use low percentile
    #   so Zone 1 is conservative: 90% of samples agree the max is ≥ this)
    # ------------------------------------------------------------------
    z1_period_min = _pct(bounds_dist.period_min_samples, z1_hi)
    z1_period_max = _pct(bounds_dist.period_max_samples, z1_lo)

    z2_period_min = _pct(bounds_dist.period_min_samples, z2_hi)
    z2_period_max = _pct(bounds_dist.period_max_samples, z2_lo)

    # ------------------------------------------------------------------
    # Duration bounds
    # ------------------------------------------------------------------
    z1_duration_min = _pct(bounds_dist.duration_min_samples, z1_hi)
    z1_duration_max = _pct(bounds_dist.duration_max_samples, z1_lo)

    z2_duration_min = _pct(bounds_dist.duration_min_samples, z2_hi)
    z2_duration_max = _pct(bounds_dist.duration_max_samples, z2_lo)

    # ------------------------------------------------------------------
    # Depth bounds
    # ------------------------------------------------------------------
    # depth_min: conservative lower bound — use low percentile so we don't
    # exclude shallow transits (err on the side of including small planets).
    # depth_max: generous upper bound — use HIGH percentile so gas giants
    # like WASP-17b (depth ~1.8%) are inside the zone.
    z1_depth_min = _pct(bounds_dist.depth_min_samples, z1_lo)
    z1_depth_max = _pct(bounds_dist.depth_max_samples, z1_hi)

    z2_depth_min = _pct(bounds_dist.depth_min_samples, z2_lo)
    z2_depth_max = _pct(bounds_dist.depth_max_samples, z2_hi)

    # ------------------------------------------------------------------
    # Build ZoneBounds objects
    # ------------------------------------------------------------------
    zone1 = ZoneBounds(
        period_min=z1_period_min,
        period_max=z1_period_max,
        duration_min=z1_duration_min,
        duration_max=z1_duration_max,
        depth_min=z1_depth_min,
        depth_max=z1_depth_max,
    )

    zone2 = ZoneBounds(
        period_min=z2_period_min,
        period_max=z2_period_max,
        duration_min=z2_duration_min,
        duration_max=z2_duration_max,
        depth_min=z2_depth_min,
        depth_max=z2_depth_max,
    )

    # Zone 3: open-ended extremes outside Zone 2
    # period < zone2.period_min  OR  period > zone2.period_max
    # Represented as two half-open intervals — we store the boundary only
    zone3 = ZoneBounds(
        period_min=None,
        period_max=z2_period_min,    # anything below Zone 2 lower bound
        duration_min=None,
        duration_max=z2_duration_min,
        depth_min=None,
        depth_max=z2_depth_min,
    )

    return zone1, zone2, zone3
