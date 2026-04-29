r"""High-level interface for J/ψ fragmentation function evolution.

Typical usage
-------------
::

    from jpsi_ff import evolve_jpsi, evolve_all_channels

    # Evolved charm FF for the dominant ³S₁^[8] channel
    z, Dc = evolve_jpsi(
        channel='3S18',
        ldme=1e-2,           # ⟨O^{J/ψ}(³S₁^[8])⟩  [GeV³]
        mu_from=3.0,         # initial scale μ₀  [GeV]
        mu_to=10.0,          # target scale        [GeV]
        alpha_s_ref=0.26,    # αs(μ₀)
        component='c',
    )

    # Sum over all four channels
    ldmes = {'1S1': 0.0, '3S18': 1e-2, '1S08': 0.0, '3PJ8': 0.0}
    z, D_total = evolve_all_channels(ldmes, 3.0, 10.0, 0.26, component='c')
"""

from __future__ import annotations

import numpy as np

from .constants import MC_GEV, MU0_OVER_MC, NF, ALPHA_EM
from .initial_conditions import initial_vector, CHANNELS
from .evolution import transfer_matrix
from .mellin import contour_nodes, invert_mellin, DEFAULT_C, DEFAULT_T, DEFAULT_N_NODES

_COMPONENT_IDX: dict[str, int] = {"gamma": 0, "c": 1, "g": 2}


def _default_z_grid() -> np.ndarray:
    return np.logspace(np.log10(0.05), np.log10(0.99), 60)


def evolve_jpsi(
    channel: str,
    ldme: float,
    mu_from: float,
    mu_to: float,
    alpha_s_ref: float,
    component: str = "c",
    z_values: np.ndarray | None = None,
    mc: float = MC_GEV,
    mu0_over_mc: float = MU0_OVER_MC,
    alpha_em: float = ALPHA_EM,
    nf: int = NF,
    n_steps: int = 50,
    mellin_c: float = DEFAULT_C,
    mellin_T: float = DEFAULT_T,
    mellin_n_nodes: int = DEFAULT_N_NODES,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Evolve the J/ψ FF for a single NRQCD channel from μ_from to μ_to.

    The algorithm:

    1. Build the Mellin contour nodes {N_k = c + it_k}.
    2. For each N_k: evaluate the initial-condition vector D̃(N_k, μ₀),
       then apply the transfer matrix E(N_k; μ₀² → μ²).
    3. Extract the requested component (γ, c, or g).
    4. Invert to z-space by numerical quadrature on the contour.

    Parameters
    ----------
    channel : str
        NRQCD channel: one of ``'1S1'``, ``'3S18'``, ``'1S08'``, ``'3PJ8'``.
    ldme : float
        Long-distance matrix element ⟨O^{J/ψ}(channel)⟩ in appropriate
        units (GeV³ for S-wave, GeV⁵ for P-wave).
    mu_from : float
        Initial scale μ₀ [GeV].  Should equal ``mu0_over_mc * mc``.
    mu_to : float
        Target scale [GeV].
    alpha_s_ref : float
        αs at ``mu_from``.
    component : {'gamma', 'c', 'g'}
        Initial parton whose FF is returned.
    z_values : array_like or None
        Momentum-fraction grid.  Defaults to 60 log-spaced points in
        [0.05, 0.99].
    mc : float
        Charm mass [GeV].
    mu0_over_mc : float
        Ratio μ₀/mc used in the NRQCD initial conditions.
    alpha_em : float
        QED coupling (fixed).
    nf : int
        Number of active flavors.
    n_steps : int
        Path-ordered integration steps inside the transfer matrix.
    mellin_c, mellin_T, mellin_n_nodes :
        Mellin inversion contour parameters.

    Returns
    -------
    z_values : numpy.ndarray, shape (n_z,)
    D_z      : numpy.ndarray, shape (n_z,)
        D_{component → J/ψ}(z, μ_to) · ldme
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got '{channel}'")
    if component not in _COMPONENT_IDX:
        raise ValueError(f"component must be 'gamma', 'c', or 'g', got '{component}'")

    comp_idx = _COMPONENT_IDX[component]
    z_arr = np.asarray(z_values if z_values is not None else _default_z_grid(), dtype=float)

    mu2_from = mu_from**2
    mu2_to = mu_to**2
    mu2_ref = (mu0_over_mc * mc) ** 2   # = mu2_from by convention

    N_nodes, weights = contour_nodes(mellin_c, mellin_T, mellin_n_nodes)

    # --- evaluate evolved Mellin FF at all contour nodes ---
    D_tilde = np.empty(len(N_nodes), dtype=complex)
    for k, N in enumerate(N_nodes):
        D0 = initial_vector(N, channel, alpha_s_ref, alpha_em, mc, mu0_over_mc)
        E = transfer_matrix(N, mu2_from, mu2_to, alpha_s_ref, mu2_ref, alpha_em, nf, n_steps)
        D_tilde[k] = (E @ D0)[comp_idx]

    D_z = invert_mellin(D_tilde, N_nodes, weights, z_arr)
    return z_arr, D_z * ldme


def evolve_all_channels(
    ldmes: dict[str, float],
    mu_from: float,
    mu_to: float,
    alpha_s_ref: float,
    component: str = "c",
    z_values: np.ndarray | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Evolve and sum over all four NRQCD channels.

        D_{total}(z, μ) = Σ_n  D_{component→J/ψ}^{(n)}(z, μ) · ⟨O(n)⟩

    Parameters
    ----------
    ldmes : dict
        ``{channel: ldme_value}`` for every channel that contributes.
        Missing channels are treated as zero.
    (remaining parameters identical to :func:`evolve_jpsi`)

    Returns
    -------
    z_values : numpy.ndarray
    D_total  : numpy.ndarray
    """
    z_arr = np.asarray(z_values if z_values is not None else _default_z_grid(), dtype=float)
    D_total = np.zeros(len(z_arr))

    for channel in CHANNELS:
        ldme = ldmes.get(channel, 0.0)
        if ldme == 0.0:
            continue
        _, D = evolve_jpsi(
            channel=channel,
            ldme=ldme,
            mu_from=mu_from,
            mu_to=mu_to,
            alpha_s_ref=alpha_s_ref,
            component=component,
            z_values=z_arr,
            **kwargs,
        )
        D_total += D

    return z_arr, D_total
