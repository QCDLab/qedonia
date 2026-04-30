r"""Mellin-space LO splitting functions and the 3×3 anomalous-dimension matrix.

Convention (from notes_jpsi.tex, §3–4):

    d D / d ln μ²  =  (αs / 2π) P̃_QCD(N) D  +  (α / 2π) P̃_QED(N) D

All P̃(N) = ∫₀¹ z^{N-1} P(z) dz are implemented below.  The ordering of the
basis vector is  (D_γ, D_c, D_g) → indices (0, 1, 2).

Splitting-function moments (eq. MPqq – MPcharm_qed of notes):

    P̃_cc(N)  = CF  [3 − 2 S₁(N) + 1/(N(N+1))]
    P̃_cg(N)  = CF  [2/(N−1) − 2/N + 1/(N+1)]
    P̃_gc(N)  = TF  [1/N − 2/(N+1) + 2/(N+2)]          ← single flavour (charm only)
    P̃_gg(N)  = 2CA [1 − S₁(N) + 1/(N(N−1)) + 1/((N+1)(N+2))] + β₀/2
    P̃_γc(N)  = ec² [1/N − 2/(N+1) + 2/(N+2)]
    P̃_cγ(N)  = ec² [2/(N−1) − 2/N + 1/(N+1)]

The QED functions (P̃_γc, P̃_cγ) already carry the factor ec² = (2/3)².
The anomalous-dimension matrix Γ is built accordingly so that both
off-diagonal QED entries carry exactly one power of ec².
"""

import numpy as np
from scipy.special import digamma

from .constants import CF, CA, TF, EC2, NF

# Euler–Mascheroni constant
_GAMMA_E: float = float(-digamma(1))


def S1(N: complex) -> complex:
    r"""Harmonic number  S₁(N) = ψ(N+1) + γ_E,  analytically continued."""
    return digamma(N + 1) + _GAMMA_E


def beta0(nf: int = NF) -> float:
    """One-loop QCD beta-function coefficient β₀ = (11 CA − 4 TF nf) / 3."""
    return (11.0 * CA - 4.0 * TF * nf) / 3.0


# ---------------------------------------------------------------------------
# QCD splitting functions
# ---------------------------------------------------------------------------


def P_cc(N: complex) -> complex:
    r"""P̃_{q→q}(N) = CF [3 − 2 S₁(N) + 1/(N(N+1))].  (eq. MPqq)"""
    return CF * (3.0 - 2.0 * S1(N) + 1.0 / (N * (N + 1.0)))


def P_cg(N: complex) -> complex:
    r"""P̃_{q→g}(N) = CF [2/(N−1) − 2/N + 1/(N+1)].  (eq. MPqg)"""
    return CF * (2.0 / (N - 1.0) - 2.0 / N + 1.0 / (N + 1.0))


def P_gc(N: complex) -> complex:
    r"""P̃_{g→c}(N) = TF [1/N − 2/(N+1) + 2/(N+2)].

    Single-flavour (charm) contribution; no nf factor.  (eq. MPgq)
    """
    return TF * (1.0 / N - 2.0 / (N + 1.0) + 2.0 / (N + 2.0))


def P_gg(N: complex, nf: int = NF) -> complex:
    r"""P̃_{g→g}(N) = 2CA [1−S₁+1/(N(N−1))+1/((N+1)(N+2))] + β₀/2.  (eq. MPgg)"""
    return (
        2.0 * CA * (1.0 - S1(N) + 1.0 / (N * (N - 1.0)) + 1.0 / ((N + 1.0) * (N + 2.0)))
        + beta0(nf) / 2.0
    )


# ---------------------------------------------------------------------------
# QED splitting functions  (both include ec² = (2/3)²)
# ---------------------------------------------------------------------------


def P_photon_c(N: complex) -> complex:
    r"""P̃_{γ→c}(N) = ec² [1/N − 2/(N+1) + 2/(N+2)].  (eq. MPgaq with eq=ec)"""
    return EC2 * (1.0 / N - 2.0 / (N + 1.0) + 2.0 / (N + 2.0))


def P_c_photon(N: complex) -> complex:
    r"""P̃_{c→γ}(N) = ec² [2/(N−1) − 2/N + 1/(N+1)].  (eq. MPqga with eq=ec)"""
    return EC2 * (2.0 / (N - 1.0) - 2.0 / N + 1.0 / (N + 1.0))


# ---------------------------------------------------------------------------
# Full 3×3 anomalous-dimension matrix
# ---------------------------------------------------------------------------


def gamma_matrix(
    N: complex,
    as_over_2pi: float,
    aem_over_2pi: float,
    nf: int = NF,
) -> np.ndarray:
    r"""Build the 3×3 anomalous-dimension matrix Γ(N, μ²).

    Basis:  row/col 0 = γ (photon),  1 = c (charm),  2 = g (gluon).

    The DGLAP equation is  dD/d ln μ² = Γ · D,  where

        Γ = ⎡     0          (α/2π) P̃_γc      0         ⎤
            ⎢ (α/2π) P̃_cγ   (αs/2π) P̃_cc   (αs/2π) P̃_cg ⎥
            ⎣     0         (αs/2π) P̃_gc   (αs/2π) P̃_gg ⎦

    Both QED entries carry exactly one power of ec² (already in P̃_γc, P̃_cγ).

    Parameters
    ----------
    N : complex
        Mellin moment.
    as_over_2pi : float
        αs(μ²) / (2π).
    aem_over_2pi : float
        α / (2π)   (fixed at charm scale by default).
    nf : int
        Number of active flavors (affects β₀ inside P̃_gg).

    Returns
    -------
    numpy.ndarray, shape (3, 3), dtype complex
    """
    Gamma = np.zeros((3, 3), dtype=complex)

    # Row 0: photon sourced by charm via QED
    Gamma[0, 1] = aem_over_2pi * P_photon_c(N)  # (α/2π) ec² [...]

    # Row 1: charm sourced by photon (QED) + charm/gluon (QCD)
    Gamma[1, 0] = aem_over_2pi * P_c_photon(N)  # (α/2π) ec² [...]
    Gamma[1, 1] = as_over_2pi * P_cc(N)
    Gamma[1, 2] = as_over_2pi * P_cg(N)

    # Row 2: gluon sourced by charm (QCD)
    Gamma[2, 1] = as_over_2pi * P_gc(N)
    Gamma[2, 2] = as_over_2pi * P_gg(N, nf)

    return Gamma
