"""LO running couplings for J/ψ fragmentation evolution.

QCD — one-loop closed form:

    αs(μ²) = αs(μ₀²) / [1 + β₀/(2π) · αs(μ₀²) · ln(μ²/μ₀²)]

QED — one-loop closed form (Landau pole direction, α increases with μ):

    1/α(μ²) = 1/α(μ₀²) − (b₀^QED / 2π) · ln(μ²/μ₀²)

where  b₀^QED = (2/3) Σ_f Nc Q_f²  summed over active quarks and leptons.
"""

from __future__ import annotations

import numpy as np

from .constants import NF, N_LEP, sum_nc_q2
from .splitting import beta0


def alpha_s(
    alpha_s_ref: float, mu2_ref: float, mu2: float | np.ndarray, nf: int = NF
) -> float | np.ndarray:
    r"""LO running strong coupling αs(μ²).

    Parameters
    ----------
    alpha_s_ref : float
        αs at the reference scale mu2_ref.
    mu2_ref : float
        Reference scale μ₀² [GeV²].
    mu2 : float or array
        Target scale(s) μ² [GeV²].
    nf : int
        Number of active flavors.

    Returns
    -------
    float or numpy.ndarray
        αs(μ²).
    """
    b0 = beta0(nf)
    lmu = np.log(np.asarray(mu2) / mu2_ref)
    return alpha_s_ref / (1.0 + (b0 / (2.0 * np.pi)) * alpha_s_ref * lmu)


def as_over_2pi(
    alpha_s_ref: float, mu2_ref: float, mu2: float | np.ndarray, nf: int = NF
) -> float | np.ndarray:
    """Return αs(μ²) / (2π), the coefficient that multiplies P̃ in DGLAP."""
    return alpha_s(alpha_s_ref, mu2_ref, mu2, nf) / (2.0 * np.pi)


def beta0_qed(nf: int = NF, n_lep: int = N_LEP) -> float:
    r"""One-loop QED β-function coefficient  b₀^QED = (2/3) Σ_f Nc Q_f².

    Parameters
    ----------
    nf : int
        Number of active quark flavors.
    n_lep : int
        Number of active unit-charge leptons (default: 3 for e, μ, τ).
    """
    return (2.0 / 3.0) * sum_nc_q2(nf, n_lep)


def alpha_em_run(
    alpha_em_ref: float,
    mu2_ref: float,
    mu2: float | np.ndarray,
    nf: int = NF,
    n_lep: int = N_LEP,
) -> float | np.ndarray:
    r"""LO running QED coupling α(μ²).

    Solves the one-loop RGE  μ² dα/dμ² = (α²/2π) b₀^QED  exactly:

        1/α(μ²) = 1/α_ref − (b₀^QED / 2π) ln(μ²/μ²_ref)

    Parameters
    ----------
    alpha_em_ref : float
        α at the reference scale ``mu2_ref``.
    mu2_ref : float
        Reference scale μ₀² [GeV²].
    mu2 : float or array
        Target scale(s) μ² [GeV²].
    nf, n_lep : int
        Active quark flavors and unit-charge leptons for b₀^QED.

    Returns
    -------
    float or numpy.ndarray
        α(μ²).
    """
    b0 = beta0_qed(nf, n_lep)
    lmu = np.log(np.asarray(mu2) / mu2_ref)
    return alpha_em_ref / (1.0 - (b0 / (2.0 * np.pi)) * alpha_em_ref * lmu)
