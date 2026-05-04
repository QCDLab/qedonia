r"""Path-ordered LO evolution operator for the (γ, c, g) 3×3 system.

Solves  dD̃/d ln μ² = Γ(N, μ²) D̃  via the iterate-exact method:

    E(N; μ₀², μ²)  =  ∏ᵢ exp[Γ(N, μ²_mid,i) · Δ(ln μ²)]

where the interval [ln μ₀², ln μ²] is split into n_steps equal steps and
Γ is evaluated at the midpoint of each step.  At LO this converges quickly;
n_steps ≈ 50 is more than sufficient for a 1-decade evolution in μ.

For each step the 3×3 matrix exponential is computed with scipy.linalg.expm,
which uses a Padé approximation and is exact for any complex matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .splitting import gamma_matrix
from .couplings import as_over_2pi, alpha_em_run
from .constants import NF, N_LEP, ALPHA_EM


def transfer_matrix(
    N: complex,
    mu2_from: float,
    mu2_to: float,
    alpha_s_ref: float,
    mu2_ref: float,
    alpha_em: float = ALPHA_EM,
    nf: int = NF,
    n_lep: int = N_LEP,
    n_steps: int = 50,
    run_alpha_em: bool = False,
) -> np.ndarray:
    r"""Compute the 3×3 transfer matrix  E(N; μ²_from → μ²_to).

        E = T exp(∫_{ln μ²_from}^{ln μ²_to} Γ(N, e^t) dt)

    Parameters
    ----------
    N : complex
        Mellin moment.
    mu2_from : float
        Initial scale μ²  [GeV²].
    mu2_to : float
        Target scale μ²   [GeV²].
    alpha_s_ref : float
        αs at the reference scale ``mu2_ref``.
    mu2_ref : float
        Reference scale for the coupling running [GeV²].
    alpha_em : float
        QED coupling at ``mu2_ref``.  Used as a fixed value when
        ``run_alpha_em=False``; used as the reference for LO running
        when ``run_alpha_em=True``.
    nf : int
        Number of active quark flavors.
    n_lep : int
        Number of active unit-charge leptons (default 3: e, μ, τ).
        Used in P̃_γγ and, when ``run_alpha_em=True``, in b₀^QED.
    n_steps : int
        Number of integration steps.  50 is accurate to < 0.1 % for a
        decade of evolution at LO.
    run_alpha_em : bool
        If True, compute α(μ²_mid) at each step via the one-loop QED RGE
        instead of keeping α fixed.

    Returns
    -------
    numpy.ndarray, shape (3, 3), dtype complex
        The transfer matrix E(N).
    """
    t_from = np.log(mu2_from)
    t_to = np.log(mu2_to)
    dt = (t_to - t_from) / n_steps

    E = np.eye(3, dtype=complex)
    for i in range(n_steps):
        t_mid = t_from + (i + 0.5) * dt
        mu2_mid = np.exp(t_mid)
        a_s_2pi = as_over_2pi(alpha_s_ref, mu2_ref, mu2_mid, nf)
        if run_alpha_em:
            aem_mid = alpha_em_run(alpha_em, mu2_ref, mu2_mid, nf, n_lep)
        else:
            aem_mid = alpha_em
        Gam = gamma_matrix(
            N,
            float(a_s_2pi),
            float(aem_mid / (2.0 * np.pi)),
            nf,
            n_lep,
        )
        E = expm(Gam * dt) @ E

    return E


def evolve_vector(
    D0: np.ndarray,
    N: complex,
    mu2_from: float,
    mu2_to: float,
    alpha_s_ref: float,
    mu2_ref: float,
    alpha_em: float = ALPHA_EM,
    nf: int = NF,
    n_lep: int = N_LEP,
    n_steps: int = 50,
    run_alpha_em: bool = False,
) -> np.ndarray:
    r"""Evolve the initial 3-vector  D̃(N, μ₀)  to scale μ².

    Parameters
    ----------
    D0 : numpy.ndarray, shape (3,)
        Initial condition vector [D̃_γ, D̃_c, D̃_g] at ``mu2_from``.
    N : complex
        Mellin moment.
    (remaining parameters identical to :func:`transfer_matrix`)

    Returns
    -------
    numpy.ndarray, shape (3,)
        Evolved vector at ``mu2_to``.
    """
    E = transfer_matrix(
        N,
        mu2_from,
        mu2_to,
        alpha_s_ref,
        mu2_ref,
        alpha_em,
        nf,
        n_lep,
        n_steps,
        run_alpha_em,
    )
    return E @ np.asarray(D0, dtype=complex)
