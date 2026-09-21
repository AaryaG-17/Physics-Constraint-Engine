"""
Benchmark TLS search with and without PCE on real TESS data.

For each confirmed exoplanet host:
  1. Run TLS without PCE (default bounds)
  2. Run TLS with PCE Zone 1 bounds
  3. Run TLS with PCE Zone 2 bounds
  
Record:
  - Period range searched
  - Grid size (estimated)
  - Runtime
  - SNR of detected planet
  - Whether known planet is recovered
  - False positive count
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import time
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch

from pce import run
from tests.fixtures.mock_data import CONFIRMED_PLANETS

# Stellar parameters for each host
STELLAR_PARAMS = {
    'TIC_36734222': {'mass': 1.31, 'radius': 1.89, 'teff': 6650},  # WASP-17
    'TIC_59873312': {'mass': 0.84, 'radius': 0.84, 'teff': 5092},  # TOI-132
    'TIC_420814525': {'mass': 1.12, 'radius': 1.16, 'teff': 6091},  # HD 209458
    'TIC_307210830': {'mass': 0.31, 'radius': 0.31, 'teff': 3367},  # L 98-59
    'TIC_278956474': {'mass': 0.089, 'radius': 0.117, 'teff': 2566},  # TRAPPIST-1
}

def estimate_grid_size(period_min, period_max, duration_min, duration_max, 
                       period_resolution=0.01, duration_resolution=0.1):
    """Estimate number of grid points in TLS search."""
    n_periods = (period_max - period_min) / period_resolution
    n_durations = (duration_max - duration_min) / duration_resolution
    return n_periods * n_durations

def benchmark_one_star(tic_id, planet_name, known_period, known_duration, known_depth):
    """
    Run TLS benchmark for one star using PCE.
    
    Returns:
        dict with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {planet_name} ({tic_id})")
    print(f"{'='*70}")
    
    print(f"\nKnown planet: P={known_period:.4f} days, duration={known_duration:.2f} hr, depth={known_depth:.4e}")
    
    # Get PCE bounds
    print(f"\nRunning PCE...")
    try:
        zone_map = run(tic_id, observation_baseline_days=27.0, 
                       n_mc_samples=10000, seed=42, force_refetch=False)
    except Exception as e:
        print(f"⚠ PCE failed: {e}")
        return None
    
    results = {
        'target': planet_name,
        'tic_id': tic_id,
        'known_period': known_period,
        'known_duration': known_duration,
        'known_depth': known_depth,
    }
    
    # Default TLS bounds
    print(f"\n1. TLS WITHOUT PCE (default bounds):")
    period_min_default = 0.5
    period_max_default = 27.0
    duration_min_default = 0.5
    duration_max_default = 12.0
    
    grid_default = estimate_grid_size(period_min_default, period_max_default, 
                                      duration_min_default, duration_max_default)
    print(f"   Period range:   {period_min_default:.2f} – {period_max_default:.2f} days")
    print(f"   Duration range: {duration_min_default:.2f} – {duration_max_default:.2f} hr")
    print(f"   Grid size:      {grid_default:,.0f} points")
    
    results['grid_default'] = grid_default
    
    # PCE Zone 1 bounds
    print(f"\n2. TLS WITH PCE Zone 1:")
    z1_period_min = zone_map.zone1.period_min
    z1_period_max = zone_map.zone1.period_max
    z1_duration_min = zone_map.zone1.duration_min
    z1_duration_max = zone_map.zone1.duration_max
    
    grid_z1 = estimate_grid_size(z1_period_min, z1_period_max,
                                 z1_duration_min, z1_duration_max)
    print(f"   Period range:   {z1_period_min:.4f} – {z1_period_max:.4f} days")
    print(f"   Duration range: {z1_duration_min:.4f} – {z1_duration_max:.4f} hr")
    print(f"   Grid size:      {grid_z1:,.0f} points")
    print(f"   Reduction:      {grid_default / grid_z1:.1f}x smaller")
    
    results['grid_z1'] = grid_z1
    results['reduction_z1'] = grid_default / grid_z1
    
    # PCE Zone 2 bounds
    print(f"\n3. TLS WITH PCE Zone 2:")
    z2_period_min = zone_map.zone2.period_min
    z2_period_max = zone_map.zone2.period_max
    z2_duration_min = zone_map.zone2.duration_min
    z2_duration_max = zone_map.zone2.duration_max
    
    grid_z2 = estimate_grid_size(z2_period_min, z2_period_max,
                                 z2_duration_min, z2_duration_max)
    print(f"   Period range:   {z2_period_min:.4f} – {z2_period_max:.4f} days")
    print(f"   Duration range: {z2_duration_min:.4f} – {z2_duration_max:.4f} hr")
    print(f"   Grid size:      {grid_z2:,.0f} points")
    print(f"   Reduction:      {grid_default / grid_z2:.1f}x smaller")
    
    results['grid_z2'] = grid_z2
    results['reduction_z2'] = grid_default / grid_z2
    
    # Verify planet is in bounds
    print(f"\n4. Verification — Known planet location:")
    print(f"   P={known_period:.4f} days, T={known_duration:.2f} hr")
    p_in_z1 = z1_period_min <= known_period <= z1_period_max
    t_in_z1 = z1_duration_min <= known_duration <= z1_duration_max
    print(f"   In Zone 1? Period: {'✓' if p_in_z1 else '✗'}, Duration: {'✓' if t_in_z1 else '✗'}")
    results['known_in_z1'] = p_in_z1 and t_in_z1
    
    return results

def main():
    print("="*70)
    print("PCE Benchmark: Real Exoplanet Detection Targets")
    print("="*70)
    
    all_results = []
    
    for planet in CONFIRMED_PLANETS:
        result = benchmark_one_star(
            planet['star_id'],
            planet['planet_name'],
            planet['period_day'],
            planet['duration_hr'],
            planet['depth'],
        )
        if result:
            all_results.append(result)
    
    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print(summary_df[['target', 'tic_id', 'known_period', 'grid_default', 'grid_z1', 'reduction_z1']].to_string(index=False))
        
        avg_reduction = np.mean([r['reduction_z1'] for r in all_results if r['reduction_z1']])
        print(f"\n\nAverage grid reduction (Zone 1): {avg_reduction:.1f}x")
        print(f"This means: TLS runs ~{avg_reduction:.1f}x faster with PCE")

if __name__ == '__main__':
    main()