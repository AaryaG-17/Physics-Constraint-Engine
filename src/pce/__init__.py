"""
pce — Physics Constraint Engine

Single entry point: pce.run()

Usage:
    from pce import run

    zone_map = run("TIC_25155310", observation_baseline_days=27.0)

    # Feed directly into TLS:
    tls.period_min = zone_map.zone1.period_min
    tls.period_max = zone_map.zone1.period_max
"""

from pce.schemas import ZoneMap
from pce.fetcher import fetch_stellar_params
from pce.sampler import run_monte_carlo
from pce.classifier import classify_zones
from pce.config import build_tls_config


def run(
    star_id: str,
    observation_baseline_days: float,
    min_transits: int = 2,
    min_transit_depth: float = 200e-6,
    n_mc_samples: int = 10_000,
    seed: int = 42,
    force_refetch: bool = False,
) -> ZoneMap:
    """
    Run PCE for one star.

    Args:
        star_id:                   TIC ID, e.g. "TIC_25155310" or "25155310"
        observation_baseline_days: Total observation window [days]
        min_transits:              Minimum transits required (2 or 3)
        min_transit_depth:         Minimum detectable depth (default 200 ppm)
        n_mc_samples:              Monte Carlo sample count (default 10 000)
        seed:                      Random seed for reproducibility (default 42)
        force_refetch:             Bypass cache and re-query catalog

    Returns:
        ZoneMap with period / duration / depth bounds in Zone 1 / 2 / 3

    Raises:
        ValueError:  invalid inputs or star not found in catalog
        RuntimeError: unexpected catalog query failure
    """

    # Stage 2: Fetch stellar parameters
    stellar_params = fetch_stellar_params(star_id, force_refetch=force_refetch)

    # Stage 3: Monte Carlo — sample uncertainties, run physics on all samples
    bounds_dist = run_monte_carlo(
        stellar_params,
        observation_baseline_days=observation_baseline_days,
        min_transits=min_transits,
        min_transit_depth=min_transit_depth,
        n_samples=n_mc_samples,
        seed=seed,
    )

    # Stage 4: Classify zones via percentile thresholds
    zone1, zone2, zone3 = classify_zones(bounds_dist)

    # Stage 5: Package for TLS
    zone_map = build_tls_config(
        zone1=zone1,
        zone2=zone2,
        zone3=zone3,
        star_id=stellar_params.tic_id,
        n_samples=n_mc_samples,
        seed=seed,
        observation_baseline_days=observation_baseline_days,
        min_transits=min_transits,
        min_transit_depth=min_transit_depth,
        catalog_source=stellar_params.catalog_source,
    )

    return zone_map
