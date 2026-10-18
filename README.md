# Physics Constraint Engine (PCE)

### Stellar-Physics-Aware Pre-filtering for Exoplanet Transit Detection

---

## Overview

The **Physics Constraint Engine (PCE)** is an astrophysics software system designed to reduce the computational search space of **Transit Least Squares (TLS)** by applying physical constraints derived from the properties of the host star.

Transit detection algorithms search for periodic dips in a star's brightness caused by a planet passing in front of the star. TLS evaluates many possible transit configurations, including orbital period, transit duration, and transit depth.

PCE asks a different question first:

> **Which of those configurations are physically plausible for this particular star?**

PCE uses stellar parameters such as mass, radius, effective temperature, luminosity, and their uncertainties to construct a physics-based map of the plausible search space.

The current implementation generates:

* **Zone 1 — Core:** Physically plausible region.
* **Zone 2 — Buffer:** Uncertainty-aware region surrounding the core.
* **Zone 3 — Excluded:** Region ruled out by the implemented physical constraints.

The current project demonstrates the PCE calculation, Monte Carlo uncertainty propagation, zone classification, and validation against known transiting planets. It also includes a separate TLS experiment using real TESS data.

**Important:** PCE and TLS are currently executed independently. Direct PCE → TLS constrained execution and measured computational reduction are future integration work.

---

# 1. Problem Statement

Transit Least Squares searches for planetary transits by testing many possible configurations of:

* **Orbital period** — the time required for a planet to complete one orbit.
* **Transit duration** — how long the planet takes to cross the stellar disk.
* **Transit depth** — the fraction of stellar light blocked by the planet.

A large portion of this parameter space may be physically impossible for a particular star.

For example:

* A planet cannot orbit arbitrarily close to its host star because tidal forces can disrupt it.
* Stellar mass and radius constrain orbital geometry.
* Orbital period constrains orbital distance through Kepler's third law.
* Orbital geometry constrains possible transit durations.
* Planet-to-star radius ratio constrains transit depth.

A conventional TLS search does not use all of these stellar-specific physical constraints to eliminate impossible regions before searching.

PCE is intended to provide that physics-based pre-filter.

---

# 2. Project Objective

The primary objective of PCE is:

> **To use stellar physics and uncertainty propagation to identify physically plausible regions of the transit-search parameter space before computationally expensive transit searches are performed.**

The intended pipeline is:

```text
Stellar Parameters
        ↓
Physics Constraint Engine
        ↓
Allowed Parameter Space
        ↓
Constrained TLS Search
        ↓
Planet Detection
```

The current implementation has reached the PCE generation and validation stage:

```text
Stellar Parameters
        ↓
PCE
        ↓
ZoneMap
        ↓
Independent TLS Experiment
        ↓
Cross-check of TLS Detection against PCE
```

---

# 3. How PCE Works

PCE takes stellar parameters and propagates their uncertainties through physical models.

The major stages are:

```text
TIC ID
  ↓
Fetch stellar parameters
  ↓
TIC / Gaia data normalization
  ↓
Monte Carlo uncertainty propagation
  ↓
Physical constraint calculations
  ↓
Zone classification
  ↓
Period-duration envelope
  ↓
ZoneMap
```

---

# 4. Physics Used

## 4.1 Roche Limit → Minimum Orbital Period

The Roche limit provides a physical constraint on how close a planet can orbit its host star before tidal forces become disruptive.

PCE uses the Roche-related constraint to establish a lower bound on physically plausible orbital periods.

Conceptually:

```text
Roche limit
     ↓
Minimum orbital distance
     ↓
Minimum orbital period
```

Periods below this boundary can be classified into the excluded region.

---

## 4.2 Kepler's Third Law

Kepler's third law connects orbital period with orbital distance and stellar mass.

Conceptually:

```text
Stellar mass + orbital period
             ↓
       Semi-major axis
```

This provides the orbital geometry required for subsequent transit-duration calculations.

---

## 4.3 Transit Geometry → Duration Constraints

Transit duration depends on:

* Stellar radius
* Planetary orbital distance
* Orbital velocity
* Transit geometry
* Impact parameter

PCE calculates physically plausible duration limits and produces a **period-duration envelope**.

This means the duration constraint is not simply one global duration range.

Instead, PCE can represent:

```text
Period  →  Allowed minimum duration
        →  Allowed maximum duration
```

The current ZoneMap contains a 128-point period-duration envelope.

---

## 4.4 Transit Depth → Planet Radius Relationship

Transit depth is related to the planet-to-star radius ratio:

```text
                 2
Depth ≈ (Rp / R★)
```

where:

* `Rp` = planet radius
* `R★` = stellar radius

This allows PCE to establish physically plausible depth limits for the target star.

---

# 5. Uncertainty Propagation

Stellar catalog values are measurements and therefore have uncertainties.

PCE does not rely only on a single deterministic set of stellar parameters.

Instead, it performs **Monte Carlo uncertainty propagation**.

For each simulation, stellar parameters are sampled from their uncertainty distributions.

The current standard demonstration configuration uses:

```text
Monte Carlo samples : 10,000
Random seed          : 42
```

The sampler supports asymmetric uncertainties using separate lower and upper uncertainties.

This is particularly important for Gaia measurements where uncertainty information is represented through asymmetric percentile bounds.

The resulting distributions are then used to determine robust zone boundaries.

---

# 6. Stellar Data Sources

PCE is designed around publicly available astronomical datasets.

## TIC-8

The **TESS Input Catalog (TIC)** provides stellar properties associated with TESS targets.

Relevant parameters include:

* TIC ID
* Effective temperature
* Stellar radius
* Stellar mass
* Luminosity
* Associated uncertainties

---

## Gaia DR3

PCE can use **Gaia DR3 astrophysical parameters** for stellar parameter acquisition and augmentation.

The implementation uses Gaia parameters such as:

* `source_id`
* `teff_gspphot`
* `radius_flame`
* `mass_flame`
* `lum_flame`
* Associated lower and upper uncertainty bounds

The current implementation can use Gaia data as the resulting catalog source when appropriate.

---

## NASA Exoplanet Archive

The NASA Exoplanet Archive provides known planetary parameters used for validation.

Relevant quantities include:

* Planet name
* TIC ID
* Orbital period
* Transit duration
* Transit depth
* Planet radius
* Transit confirmation information

---

## TESS Light Curves

Real TESS light curves are used by the TLS experiment.

They provide the time-series flux data required by TLS.

---

## Kepler Light Curves

Kepler light curves are part of the broader project data strategy and can provide longer observational baselines for transit studies.

---

# 7. System Architecture

### Current implemented architecture

```text
                 ┌──────────────────────┐
                 │   Stellar Catalogs   │
                 │    TIC / Gaia DR3    │
                 └──────────┬───────────┘
```
