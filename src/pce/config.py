"""
config.py — TLS Configuration Builder (Stage 5)

Packages Zone 1/2/3 bounds into a ZoneMap — the final PCE output
ready for consumption by the Transit Least Squares (TLS) algorithm.

Public API:
    build_tls_config(zone1, zone2, zone3, star_id, **metadata) -> ZoneMap

Usage after build:
    zone_map = build_tls_config(...)
    tls.period_min = zone_map.zone1.period_min
    tls.period_max = zone_map.zone1.period_max
"""

from datetime import datetime, timezone
from pce.schemas import ZoneBounds, ZoneMap


def build_tls_config(
    zone1: ZoneBounds,
    zone2: ZoneBounds,
    zone3: ZoneBounds,
    star_id: str,
    n_samples: int,
    seed: int,
    observation_baseline_days: float,
    min_transits: int,
    min_transit_depth: float,
    catalog_source: str,
) -> ZoneMap:
    """
    Package zone bounds and run metadata into a ZoneMap for TLS.

    Args:
        zone1:                     High-confidence zone (≥90% MC agreement)
        zone2:                     Uncertain zone (50–90% MC agreement)
        zone3:                     Excluded zone (<50% or implausible)
        star_id:                   TIC identifier
        n_samples:                 MC sample count used
        seed:                      Random seed used
        observation_baseline_days: Observation window [days]
        min_transits:              Minimum transits required
        min_transit_depth:         Minimum detectable depth [dimensionless]
        catalog_source:            'TIC-8' or 'Gaia'

    Returns:
        ZoneMap — validated Pydantic model, ready for TLS
    """
    return ZoneMap(
        star_id=star_id,
        zone1=zone1,
        zone2=zone2,
        zone3=zone3,
        n_samples=n_samples,
        seed=seed,
        observation_baseline_days=observation_baseline_days,
        min_transits=min_transits,
        min_transit_depth=min_transit_depth,
        catalog_source=catalog_source,
        generated_at=datetime.now(timezone.utc),
    )
