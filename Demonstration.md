# Physics Constraint Engine (PCE)

## Demonstration — Exact Commands

## PCE Demonstration

### 1. Enter the PCE Environment

```bash
cd ~/Projects/Physics-Constraint-Engine
source .venv/bin/activate
```

Activates the PCE project's Python virtual environment.

### 2. Configure the Python Source Path

```bash
export PYTHONPATH="$PWD/src"
```

Adds the project's `src` directory to Python's module search path.

### 3. Inspect the Stellar Data Fetcher

```bash
python -c "import inspect; from pce.fetcher import fetch_stellar_params; print(inspect.signature(fetch_stellar_params))"
```

Displays the interface and parameters of the stellar-parameter fetching function.

### 4. Fetch Stellar Parameters

```bash
python -c "from pce.fetcher import fetch_stellar_params; print(fetch_stellar_params('TIC_420814525'))"
```

Fetches the stellar parameters and uncertainties for TIC 420814525.

### 5. Inspect the Physics Layer

```bash
grep -n "^def " src/pce/physics.py
```

Lists the physics functions implemented by PCE.

### 6. Inspect the Monte Carlo Sampler

```bash
grep -n "^def " src/pce/sampler.py
```

Lists the functions used for uncertainty sampling and Monte Carlo propagation.

### 7. Inspect the Zone Classifier

```bash
grep -n "^def " src/pce/classifier.py
```

Lists the function responsible for classifying the calculated parameter space into zones.

### 8. Inspect the Main PCE Interface

```bash
python -c "import inspect; from pce import run; print(inspect.signature(run))"
```

Displays the interface and configurable parameters of the main PCE pipeline.

### 9. Run the Complete PCE Pipeline

```bash
python -c "from pce import run; z=run('TIC_420814525', observation_baseline_days=27, min_transits=2, min_transit_depth=0.0002, n_mc_samples=10000, seed=42); print(z)"
```

Runs PCE for TIC 420814525 and generates the physics-based `ZoneMap`.

### 10. Validate Against Confirmed Planets

```bash
python validation/validate_confirmed_planets.py --help
```

Runs the project's confirmed-planet validation against its fixture set.

---

# Standalone TLS Demonstration

### 11. Switch to the TLS Environment

```bash
deactivate
source .venv-tls/bin/activate
python --version
```

Switches from the PCE environment to the dedicated TLS environment and verifies the Python version.

### 12. List the Real TESS Light Curves

```bash
ls data/real_lightcurves/
```

Lists the real TESS light-curve files available for the TLS experiments.

### 13. Inspect the TLS Baseline Experiment

```bash
sed -n '1,240p' experiments/run_tls_baseline.py
```

Displays the existing TLS baseline experiment before running it.

### 14. Run the Standalone TLS Baseline

```bash
python experiments/run_tls_baseline.py
```

Runs TLS independently on the real TESS light curve and reports its search and detection statistics.

---

# Demonstration Flow

```text
PCE:
Stellar Data
    ↓
Physics
    ↓
Monte Carlo
    ↓
Zone Classification
    ↓
ZoneMap
    ↓
Confirmed-Planet Validation

TLS:
Real TESS Light Curve
    ↓
Standalone TLS Search
    ↓
Transit Detection
```

**Note:** The TLS section is intentionally independent of PCE. Direct PCE → TLS integration and measurement of computational reduction are not part of this demonstration yet.