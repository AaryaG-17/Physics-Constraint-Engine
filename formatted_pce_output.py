from pce import run

z = run(
    "TIC_420814525",
    observation_baseline_days=27,
    min_transits=2,
    min_transit_depth=0.0002,
    n_mc_samples=10000,
    seed=42,
)

print()
print("=" * 55)
print("        PCE RESULT — TIC 420814525")
print("=" * 55)

print(f"Star ID              : {z.star_id}")
print(f"Monte Carlo samples  : {z.n_samples}")
print(f"Random seed          : {z.seed}")
print(f"Catalog source       : {z.catalog_source}")

print()
print("ZONE 1 — CORE ALLOWED REGION")
print("-" * 55)
print(
    f"Period range         : "
    f"{z.zone1.period_min:.6f} – {z.zone1.period_max:.6f} days"
)
print(
    f"Duration range       : "
    f"{z.zone1.duration_min:.6f} – {z.zone1.duration_max:.6f}"
)
print(
    f"Depth range          : "
    f"{z.zone1.depth_min:.6f} – {z.zone1.depth_max:.6f}"
)

print()
print("ZONE 2 — BUFFER REGION")
print("-" * 55)
print(
    f"Period range         : "
    f"{z.zone2.period_min:.6f} – {z.zone2.period_max:.6f} days"
)
print(
    f"Duration range       : "
    f"{z.zone2.duration_min:.6f} – {z.zone2.duration_max:.6f}"
)
print(
    f"Depth range          : "
    f"{z.zone2.depth_min:.6f} – {z.zone2.depth_max:.6f}"
)

print()
print("ZONE 3 — EXCLUDED REGION")
print("-" * 55)
print(f"Period <=            : {z.zone3.period_max:.6f} days")
print(f"Duration <=          : {z.zone3.duration_max:.6f}")
print(f"Depth <=             : {z.zone3.depth_max:.6f}")

envelope = z.zone1.period_duration_envelope

print()
print("PERIOD-DURATION ENVELOPE")
print("-" * 55)
print(f"Number of period points : {len(envelope.periods)}")
print(
    f"Period span             : "
    f"{envelope.periods[0]:.6f} – {envelope.periods[-1]:.6f} days"
)

print("=" * 55)