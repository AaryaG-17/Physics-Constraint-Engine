"""
validate_confirmed_planets.py — Stage 6: Validation Against Confirmed Planets

Tests PCE against known confirmed exoplanet host stars.
For each star, runs the full pipeline and checks whether the actual
planet's period/duration/depth falls within Zone 1 or Zone 2.

Success criteria:
    Zone 1 hit: actual planet parameters fall inside Zone 1 bounds → best case
    Zone 2 hit: fall inside Zone 2 but outside Zone 1 → acceptable
    Miss:       fall outside Zone 2 → pipeline failure for this star

Target: >95% of confirmed planets should be Zone 1 or Zone 2 hits.

Usage:
    python validation/validate_confirmed_planets.py

Output:
    - Console summary (success rate, misses)
    - data/validation_results.csv (detailed per-star results)
"""

import sys
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, "src")
sys.path.insert(0, ".")   # makes tests/ importable when run from pce_project/

from pce.schemas import ZoneMap, ZoneBounds
from pce import run
from tests.fixtures.mock_data import CONFIRMED_PLANETS


# ---------------------------------------------------------------------------
# Zone membership check
# ---------------------------------------------------------------------------

def _in_zone(value: float, zone: ZoneBounds, field_min: str, field_max: str) -> bool:
    """
    Check whether value falls within zone's [field_min, field_max] bounds.
    None bounds are treated as open-ended (no constraint in that direction).
    """
    lo = getattr(zone, field_min)
    hi = getattr(zone, field_max)
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def check_planet_in_zones(
    period_day: float,
    duration_hr: float,
    depth: float,
    zone_map: ZoneMap,
) -> dict:
    """
    Check whether actual planet parameters fall in Zone 1, Zone 2, or neither.

    Returns:
        dict with keys:
            period_zone:   1, 2, or 0 (miss)
            duration_zone: 1, 2, or 0
            depth_zone:    1, 2, or 0
            overall:       'zone1' | 'zone2' | 'miss'
                           zone1 = all three in Zone 1
                           zone2 = all three in Zone 1 or Zone 2
                           miss  = at least one parameter outside Zone 2
    """
    def _classify(val, min_field, max_field):
        if _in_zone(val, zone_map.zone1, min_field, max_field):
            return 1
        if _in_zone(val, zone_map.zone2, min_field, max_field):
            return 2
        return 0

    period_zone   = _classify(period_day,   "period_min",   "period_max")
    duration_zone = _classify(duration_hr,  "duration_min", "duration_max")
    depth_zone    = _classify(depth,        "depth_min",    "depth_max")

    if all(z == 1 for z in [period_zone, duration_zone, depth_zone]):
        overall = "zone1"
    elif all(z in (1, 2) for z in [period_zone, duration_zone, depth_zone]):
        overall = "zone2"
    else:
        overall = "miss"

    return {
        "period_zone":   period_zone,
        "duration_zone": duration_zone,
        "depth_zone":    depth_zone,
        "overall":       overall,
    }


# ---------------------------------------------------------------------------
# Single-star validation
# ---------------------------------------------------------------------------

def validate_one_star(
    star_id: str,
    planet_name: str,
    period_day: float,
    duration_hr: float,
    depth: float,
    observation_baseline_days: float = 27.0,
    n_mc_samples: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Run the full PCE pipeline for one star and check the planet against zones.

    Returns:
        dict with star_id, planet_name, zone results, and ZoneMap bounds
    """
    try:
        zone_map = run(
            star_id,
            observation_baseline_days=observation_baseline_days,
            n_mc_samples=n_mc_samples,
            seed=seed,
        )
        result = check_planet_in_zones(period_day, duration_hr, depth, zone_map)

        return {
            "star_id":       star_id,
            "planet_name":   planet_name,
            "period_day":    period_day,
            "duration_hr":   duration_hr,
            "depth":         depth,
            "overall":       result["overall"],
            "period_zone":   result["period_zone"],
            "duration_zone": result["duration_zone"],
            "depth_zone":    result["depth_zone"],
            "z1_period_min": zone_map.zone1.period_min,
            "z1_period_max": zone_map.zone1.period_max,
            "z1_duration_min": zone_map.zone1.duration_min,
            "z1_duration_max": zone_map.zone1.duration_max,
            "z1_depth_min":  zone_map.zone1.depth_min,
            "z1_depth_max":  zone_map.zone1.depth_max,
            "z2_period_min": zone_map.zone2.period_min,
            "z2_period_max": zone_map.zone2.period_max,
            "catalog_source": zone_map.catalog_source,
            "error":         None,
        }

    except Exception as exc:
        return {
            "star_id":     star_id,
            "planet_name": planet_name,
            "period_day":  period_day,
            "duration_hr": duration_hr,
            "depth":       depth,
            "overall":     "error",
            "error":       str(exc),
        }


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def validate_against_confirmed(
    planets: list = None,
    observation_baseline_days: float = 27.0,
    n_mc_samples: int = 10_000,
    seed: int = 42,
    output_csv: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Validate PCE against a list of confirmed planet host stars.

    Args:
        planets:                   List of planet dicts (default: CONFIRMED_PLANETS)
        observation_baseline_days: Observation baseline for all stars [days]
        n_mc_samples:              MC samples per star
        seed:                      Random seed
        output_csv:                Path to write detailed CSV results (optional)
        verbose:                   Print per-star progress

    Returns:
        Summary dict:
            n_total:    total planets tested
            n_zone1:    planets where all params hit Zone 1
            n_zone2:    planets where all params hit Zone 1 or Zone 2
            n_miss:     planets outside Zone 2 for at least one param
            n_error:    stars that raised an exception
            success_rate: (n_zone1 + n_zone2) / n_total
            results:    list of per-star result dicts
    """
    if planets is None:
        planets = CONFIRMED_PLANETS

    results = []
    n_zone1 = n_zone2 = n_miss = n_error = 0

    for planet in planets:
        if verbose:
            print(f"  Validating {planet['planet_name']} ({planet['star_id']})...", end=" ")

        r = validate_one_star(
            star_id=planet["star_id"],
            planet_name=planet["planet_name"],
            period_day=planet["period_day"],
            duration_hr=planet["duration_hr"],
            depth=planet["depth"],
            observation_baseline_days=observation_baseline_days,
            n_mc_samples=n_mc_samples,
            seed=seed,
        )
        results.append(r)

        if r["overall"] == "zone1":
            n_zone1 += 1
            if verbose: print("ZONE 1 ✓")
        elif r["overall"] == "zone2":
            n_zone2 += 1
            if verbose: print("ZONE 2 ✓")
        elif r["overall"] == "error":
            n_error += 1
            if verbose: print(f"ERROR — {r['error']}")
        else:
            n_miss += 1
            if verbose:
                print(
                    f"MISS — period_zone={r.get('period_zone')}, "
                    f"duration_zone={r.get('duration_zone')}, "
                    f"depth_zone={r.get('depth_zone')}"
                )

    n_total = len(planets)
    n_counted = n_zone1 + n_zone2 + n_miss  # errors excluded from rate
    success_rate = (n_zone1 + n_zone2) / n_counted if n_counted > 0 else 0.0

    summary = {
        "n_total":      n_total,
        "n_zone1":      n_zone1,
        "n_zone2":      n_zone2,
        "n_miss":       n_miss,
        "n_error":      n_error,
        "success_rate": success_rate,
        "results":      results,
    }

    if verbose:
        print()
        print("=" * 50)
        print(f"Results: {n_total} planets tested")
        print(f"  Zone 1 hits : {n_zone1}")
        print(f"  Zone 2 hits : {n_zone2}")
        print(f"  Misses      : {n_miss}")
        print(f"  Errors      : {n_error}")
        print(f"  Success rate: {success_rate * 100:.1f}%")
        print("=" * 50)

    if output_csv is not None:
        _write_csv(results, output_csv)
        if verbose:
            print(f"Detailed results written to {output_csv}")

    return summary


def _write_csv(results: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    fieldnames = [k for k in results[0].keys()]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("PCE Validation — Confirmed Planets")
    print(f"Testing {len(CONFIRMED_PLANETS)} planets from fixture set")
    print()

    output_path = Path("data/validation_results.csv")
    validate_against_confirmed(
        observation_baseline_days=27.0,
        n_mc_samples=10_000,
        seed=42,
        output_csv=output_path,
        verbose=True,
    )
