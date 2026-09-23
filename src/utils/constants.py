"""
constants.py — Physical Constants for PCE

All constants are astropy Quantity objects with explicit units.
Sources cited per constant.

Do NOT import raw floats from here — always use the Quantity form
so units propagate correctly through physics.py.
"""

from astropy import constants as const
from astropy import units as u

# ---------------------------------------------------------------------------
# Gravitational constant
# Source: NIST CODATA 2018 — via astropy.constants.G
# ---------------------------------------------------------------------------
G = const.G.to(u.m**3 / (u.kg * u.s**2))

# ---------------------------------------------------------------------------
# Solar parameters
# Source: IAU 2015 nominal solar values (resolution B3)
# ---------------------------------------------------------------------------
M_SUN = const.M_sun.to(u.kg)           # Solar mass
R_SUN = const.R_sun.to(u.m)            # Solar radius
L_SUN = const.L_sun.to(u.W)            # Solar luminosity

# ---------------------------------------------------------------------------
# Roche limit period formula — Rappaport et al. (2013), ApJL 773, L15
# Simplified fluid-body formulation used throughout PCE:
#     P_Roche = ROCHE_PERIOD_PREFACTOR_HR * (rho_planet / 1 g/cm^3)^(-0.5)
# This constant (12.6 hr) is the pre-factor from combining the fluid-body
# Roche distance with Kepler's third law, which cancels stellar mass and
# radius entirely. This is what physics.py uses directly (_ROCHE_PREFACTOR).
# The intermediate Roche distance constant (2.44, fluid body) is NOT used
# in PCE — the period formula is used directly. Do not add C_ROCHE back.
# ---------------------------------------------------------------------------
ROCHE_PERIOD_PREFACTOR_HR = 12.6   # hours — Rappaport et al. (2013)

# ---------------------------------------------------------------------------
# Planet density bounds [kg/m^3]
# Used to bracket the Roche limit over plausible planet compositions.
# Source:
#   Lower bound — Saturn-like (low-density gas giant)
#     Seiff et al. (1998), J. Geophys. Res., 103(E10), 22857
#     Saturn mean density ~ 687 kg/m^3; use 500 as conservative floor
#   Upper bound — Iron-rich rocky planet
#     Fortney et al. (2007), ApJ, 659, 1661
#     Maximum density for rocky planets ~ 10 000 kg/m^3
# ---------------------------------------------------------------------------
PLANET_DENSITY_MIN = 30.0 * u.kg / u.m**3      # 0.03 g/cm³ — conservative ~1σ lower envelope
                                                 # below Kepler-51d (0.038±0.009 g/cm³) and
                                                 # TOI-791b (0.038±0.008 g/cm³).
                                                 # NOT itself an observed density.
                                                 # Source: JWST NIRSpec-PRISM (AJ, 2026);
                                                 #         ASTEP confirmation arXiv:2606.30016
PLANET_DENSITY_MAX = 14_100.0 * u.kg / u.m**3  # 14.1 g/cm³ — TOI-4603b (Khandelwal et al.
                                                 # 2023, A&A 675, A45). Densest confirmed
                                                 # exoplanet. Caveat: sits at the ~13 MJup
                                                 # planet/brown-dwarf boundary.

# ---------------------------------------------------------------------------
# Default minimum detectable transit depth
# 200 ppm — conservative floor for TESS photometric precision.
# Source: Sullivan et al. (2015), ApJ, 809, 77 — TESS yield simulations
# ---------------------------------------------------------------------------
MIN_TRANSIT_DEPTH_PPM = 200.0                           # parts per million
MIN_TRANSIT_DEPTH = MIN_TRANSIT_DEPTH_PPM * 1e-6        # dimensionless

# ---------------------------------------------------------------------------
# Minimum number of transits required for detection
# Default = 2 (TLS standard; 3 is more conservative)
# Source: Hippke & Heller (2019), A&A, 623, A39 — TLS paper
# ---------------------------------------------------------------------------
MIN_TRANSITS_DEFAULT = 2

# ---------------------------------------------------------------------------
# Monte Carlo defaults
# ---------------------------------------------------------------------------
N_MC_SAMPLES_DEFAULT = 10_000
MC_SEED_DEFAULT = 42

# ---------------------------------------------------------------------------
# Zone percentile thresholds
# Zone 1: >=90% of MC samples agree  -> high confidence
# Zone 2:  50-90% agree              -> uncertain edges
# Zone 3: <50% agree                 -> excluded
# ---------------------------------------------------------------------------
ZONE1_PERCENTILE = 0.90
ZONE2_PERCENTILE = 0.50
