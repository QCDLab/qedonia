"""Physical constants and color factors for J/ψ fragmentation evolution."""

# QCD color factors  (SU(3), Nc = 3)
CF: float = 4.0 / 3.0
CA: float = 3.0
TF: float = 0.5
NC: int = 3

# Charm quark electric charge (in units of |e|)
EC: float = 2.0 / 3.0
EC2: float = EC**2

# Default physics parameters
MC_GEV: float = 1.5  # charm quark mass [GeV]
MU0_OVER_MC: float = 2.0  # μ₀ / mc  (initial fragmentation scale)
NF: int = 4  # active flavors at charm scale: u, d, s, c
ALPHA_EM: float = 1.0 / 133.0  # QED coupling (fixed at charm scale)
