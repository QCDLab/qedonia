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
NF: int = 4  # active QCD flavors at charm scale: u, d, s, c
ALPHA_EM: float = 1.0 / 133.0  # QED reference coupling α(μ₀ = 3 GeV)

# ---------------------------------------------------------------------------
# QED fermion content — used for the running coupling and P̃_γγ
# ---------------------------------------------------------------------------

N_LEP: int = 3  # active leptons in the GeV range: e, μ, τ

# Nc * Q_f^2  for each quark flavor, ordered by mass (u, d, s, c, b)
_QUARK_NCQ2: tuple[float, ...] = (
    NC * (2.0 / 3.0) ** 2,  # u
    NC * (1.0 / 3.0) ** 2,  # d
    NC * (1.0 / 3.0) ** 2,  # s
    NC * (2.0 / 3.0) ** 2,  # c
    NC * (1.0 / 3.0) ** 2,  # b
)


def sum_nc_q2(nf: int = NF, n_lep: int = N_LEP) -> float:
    """Return Σ_f Nc Q_f² summed over *nf* quarks and *n_lep* unit-charge leptons."""
    return sum(_QUARK_NCQ2[:nf]) + float(n_lep)
