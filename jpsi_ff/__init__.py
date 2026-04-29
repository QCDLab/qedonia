"""jpsi_ff — LO coupled QCD⊗QED evolution of J/ψ fragmentation functions.

The module evolves the three fragmentation functions D_γ, D_c, D_g to J/ψ
jointly via the 3×3 anomalous-dimension matrix at leading order in both
αs and α, using NRQCD initial conditions at μ₀ = 2mc.

Four NRQCD channels are supported:

    '1S1'   ³S₁^[1]   color-singlet  (Braaten–Yuan IC for c; vanishes for g)
    '3S18'  ³S₁^[8]   color-octet    (dominant δ(1−z) IC for c and g)
    '1S08'  ¹S₀^[8]   color-octet
    '3PJ8'  ³P_J^[8]  color-octet    (J=0,1,2 summed)

Quick start::

    from jpsi_ff import evolve_jpsi, evolve_all_channels

    z, Dc = evolve_jpsi(
        channel='3S18', ldme=1e-2,
        mu_from=3.0, mu_to=10.0, alpha_s_ref=0.26,
        component='c',
    )
"""

from .runner import evolve_jpsi, evolve_all_channels
from .initial_conditions import initial_vector, CHANNELS, F_gamma, braaten_yuan_moment
from .splitting import (
    gamma_matrix,
    P_cc,
    P_cg,
    P_gc,
    P_gg,
    P_photon_c,
    P_c_photon,
    S1,
    beta0,
)
from .couplings import alpha_s, as_over_2pi
from .evolution import transfer_matrix, evolve_vector
from .mellin import contour_nodes, invert_mellin

__all__ = [
    # high-level API
    "evolve_jpsi",
    "evolve_all_channels",
    # initial conditions
    "initial_vector",
    "CHANNELS",
    "F_gamma",
    "braaten_yuan_moment",
    # splitting functions
    "gamma_matrix",
    "P_cc",
    "P_cg",
    "P_gc",
    "P_gg",
    "P_photon_c",
    "P_c_photon",
    "S1",
    "beta0",
    # couplings
    "alpha_s",
    "as_over_2pi",
    # evolution
    "transfer_matrix",
    "evolve_vector",
    # Mellin
    "contour_nodes",
    "invert_mellin",
]
