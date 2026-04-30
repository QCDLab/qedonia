"""LO QCD running coupling for J/ψ fragmentation evolution.

At LO the closed-form solution of the RGE is

    αs(μ²) = αs(μ₀²) / [1 + β₀/(2π) · αs(μ₀²) · ln(μ²/μ₀²)]

The QED coupling α is kept fixed at the charm scale throughout
(suppression α/αs ~ 1/10 makes running negligible at LO).
"""

from __future__ import annotations

import numpy as np

from .constants import NF
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
