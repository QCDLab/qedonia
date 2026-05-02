r"""NRQCD initial conditions at μ₀ = mu0_over_mc · mc in Mellin space.

The fragmentation function at the initial scale is decomposed as (eq. nrqcd_sum):

    D̃_{i→J/ψ}(N, μ₀) = Σ_n  d̃_{i→cc̄[n]}(N, μ₀) · ⟨O^{J/ψ}(n)⟩

The four NRQCD channels are:

    '1S1'  :  ³S₁^[1]   color-singlet  (Braaten–Yuan)
    '3S18' :  ³S₁^[8]   color-octet    (dominant, δ(1−z) IC)
    '1S08' :  ¹S₀^[8]   color-octet
    '3PJ8' :  ³P_J^[8]  color-octet    (J = 0,1,2 summed)

Three universal building blocks cover all channels (eqs. Mdelta–Mphotshape):

    ∫₀¹ z^{N-1} δ(1−z) dz                         = 1
    ∫₀¹ z^{N-1} z(1−z) dz                          = 1/[(N+1)(N+2)]
    ∫₀¹ z^{N-1} [z(2−2z+z²)/(1−z)]_reg dz         = F_γ(N)

where F_γ(N) = 1 − S₁(N) + 1/(N+1) − 1/(N+2) + ln(μ₀²/mc²).

The Braaten–Yuan moment B(N) for the charm color-singlet channel has no
closed form and is evaluated by numerical quadrature (eq. BYmoment).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma

from .constants import EC, MC_GEV, MU0_OVER_MC

_GAMMA_E: float = float(-digamma(1))

CHANNELS: tuple[str, ...] = ("1S1", "3S18", "1S08", "3PJ8")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _S1(N: complex) -> complex:
    return digamma(N + 1) + _GAMMA_E


def F_gamma(N: complex, mu0_over_mc: float = MU0_OVER_MC) -> complex:
    r"""Photon-shape Mellin moment  F_γ(N).

    F_γ(N) = 1 − S₁(N) + 1/(N+1) − 1/(N+2) + ln(μ₀²/mc²)  (eq. Fgamma)

    This is the Mellin transform of z(2−2z+z²)/(1−z) regularised via the
    plus-prescription with endpoint term ln(μ₀²/mc²).
    """
    ln_ratio = np.log(mu0_over_mc**2)  # = ln(4) ≈ 1.386 for mu0 = 2mc
    return 1.0 - _S1(N) + 1.0 / (N + 1.0) - 1.0 / (N + 2.0) + ln_ratio


def _zz_moment(N: complex) -> complex:
    """Mellin moment of z(1−z): 1/[(N+1)(N+2)]."""
    return 1.0 / ((N + 1.0) * (N + 2.0))


# ---------------------------------------------------------------------------
# Braaten–Yuan integral  B(N)  (eq. BYmoment)
# ---------------------------------------------------------------------------


def _by_integrand_re(z: float, N_re: float, N_im: float) -> float:
    if z <= 0.0:
        return 0.0
    poly = 16.0 - 32.0 * z + 72.0 * z**2 - 32.0 * z**3 + 5.0 * z**4
    base = z * (1.0 - z) ** 2 / (2.0 - z) ** 6 * poly
    ln_z = np.log(z)
    amp = np.exp((N_re - 1.0) * ln_z) * base
    return float(amp * np.cos(N_im * ln_z))


def _by_integrand_im(z: float, N_re: float, N_im: float) -> float:
    if z <= 0.0:
        return 0.0
    poly = 16.0 - 32.0 * z + 72.0 * z**2 - 32.0 * z**3 + 5.0 * z**4
    base = z * (1.0 - z) ** 2 / (2.0 - z) ** 6 * poly
    ln_z = np.log(z)
    amp = np.exp((N_re - 1.0) * ln_z) * base
    return float(amp * np.sin(N_im * ln_z))


def braaten_yuan_moment(N: complex, tol: float = 1e-9) -> complex:
    r"""Numerical Mellin moment B(N) for the Braaten–Yuan charm color-singlet IC.

        B(N) = ∫₀¹ z^{N-1} · z(1−z)² (16 − 32z + 72z² − 32z³ + 5z⁴) / (2−z)⁶ dz

    The integrand is smooth on (0,1]; no endpoint treatment is required.

    Parameters
    ----------
    N : complex
        Mellin moment.
    tol : float
        Absolute and relative tolerance for quad.

    Returns
    -------
    complex
    """
    N_re = N.real
    N_im = N.imag

    re_val, _ = quad(
        _by_integrand_re,
        0.0,
        1.0,
        args=(N_re, N_im),
        limit=200,
        epsabs=tol,
        epsrel=tol,
    )
    if abs(N_im) < 1e-14:
        return complex(re_val)

    im_val, _ = quad(
        _by_integrand_im,
        0.0,
        1.0,
        args=(N_re, N_im),
        limit=200,
        epsabs=tol,
        epsrel=tol,
    )
    return complex(re_val, im_val)


# ---------------------------------------------------------------------------
# Short-distance coefficients  d̃_{i→cc̄[n]}(N)
# Units: [d̃] · [⟨O⟩] = dimensionless FF,  so [d̃] = 1/[⟨O⟩].
# ⟨O^{J/ψ}(³S₁^[1,8])⟩  in GeV³,  ⟨O^{J/ψ}(³P_J^[8])⟩  in GeV⁵.
# ---------------------------------------------------------------------------


def d_photon(
    N: complex,
    channel: str,
    alpha_s_ref: float,
    alpha_em: float,
    mc: float = MC_GEV,
    mu0_over_mc: float = MU0_OVER_MC,
) -> complex:
    r"""d̃_{γ→cc̄[channel]}(N) — photon short-distance coefficient.

    Parameters
    ----------
    N : complex
        Mellin moment.
    channel : str
        One of ``'1S1'``, ``'3S18'``, ``'1S08'``, ``'3PJ8'``.
    alpha_s_ref : float
        αs(μ₀).
    alpha_em : float
        α (QED coupling, fixed).
    mc : float
        Charm mass [GeV].
    mu0_over_mc : float
        Ratio μ₀/mc.

    Returns
    -------
    complex
    """
    F = F_gamma(N, mu0_over_mc)
    zz = _zz_moment(N)

    if channel == "1S1":
        # eq. Mga_1S1: (32π α² ec⁴)/(27 mc³) · F_γ(N)
        return (32.0 * np.pi * alpha_em**2 * EC**4) / (27.0 * mc**3) * F
    elif channel == "3S18":
        # eq. Mga_3S18: (4π α² ec⁴)/(3 mc³) · F_γ(N)
        return (4.0 * np.pi * alpha_em**2 * EC**4) / (3.0 * mc**3) * F
    elif channel == "1S08":
        # eq. Mga_1S08: (π α² ec⁴)/mc³ · 1/[(N+1)(N+2)]
        return (np.pi * alpha_em**2 * EC**4) / mc**3 * zz
    elif channel == "3PJ8":
        # eq. Mga_3PJ8: (π α² ec⁴)/mc⁵ · 1/[(N+1)(N+2)]
        return (np.pi * alpha_em**2 * EC**4) / mc**5 * zz
    else:
        raise ValueError(f"Unknown channel '{channel}'. Choose from {CHANNELS}.")


def d_charm(
    N: complex,
    channel: str,
    alpha_s_ref: float,
    alpha_em: float,
    mc: float = MC_GEV,
    mu0_over_mc: float = MU0_OVER_MC,
) -> complex:
    r"""d̃_{c→cc̄[channel]}(N) — charm quark short-distance coefficient."""
    zz = _zz_moment(N)

    if channel == "1S1":
        # eq. Mc_1S1: (16 αs²)/(729 mc³) · B(N)   (Braaten–Yuan / Cho-Leibovich)
        B = braaten_yuan_moment(N)
        return (16.0 * alpha_s_ref**2) / (729.0 * mc**3) * B
    elif channel == "3S18":
        # eq. Mc_3S18: (π αs)/(6 mc³)   (constant in N, from δ(1-z))
        return (np.pi * alpha_s_ref) / (6.0 * mc**3) + 0j
    elif channel == "1S08":
        # eq. Mc_1S08: αs²/mc³ · 1/[(N+1)(N+2)]
        return alpha_s_ref**2 / mc**3 * zz
    elif channel == "3PJ8":
        # eq. Mc_3PJ8: αs²/mc⁵ · 1/[(N+1)(N+2)]
        return alpha_s_ref**2 / mc**5 * zz
    else:
        raise ValueError(f"Unknown channel '{channel}'. Choose from {CHANNELS}.")


def d_gluon(
    N: complex,
    channel: str,
    alpha_s_ref: float,
    alpha_em: float,
    mc: float = MC_GEV,
    mu0_over_mc: float = MU0_OVER_MC,
) -> complex:
    r"""d̃_{g→cc̄[channel]}(N) — gluon short-distance coefficient."""
    zz = _zz_moment(N)

    if channel == "1S1":
        # eq. Mg_1S1: vanishes at LO
        return 0.0 + 0j
    elif channel == "3S18":
        # eq. Mg_3S18: (π αs)/(24 mc³)   (ratio 1/4 vs charm, from color factor)
        return (np.pi * alpha_s_ref) / (24.0 * mc**3) + 0j
    elif channel == "1S08":
        # eq. Mg_1S08: αs²/mc³ · 1/[(N+1)(N+2)]
        return alpha_s_ref**2 / mc**3 * zz
    elif channel == "3PJ8":
        # eq. Mg_3PJ8: αs²/mc⁵ · 1/[(N+1)(N+2)]
        return alpha_s_ref**2 / mc**5 * zz
    else:
        raise ValueError(f"Unknown channel '{channel}'. Choose from {CHANNELS}.")


def initial_vector(
    N: complex,
    channel: str,
    alpha_s_ref: float,
    alpha_em: float,
    mc: float = MC_GEV,
    mu0_over_mc: float = MU0_OVER_MC,
) -> np.ndarray:
    r"""Build the initial 3-vector  D̃(N, μ₀)  for a given NRQCD channel.

    The returned vector does NOT include the LDME ⟨O^{J/ψ}(n)⟩; multiply
    it externally.

    Returns
    -------
    numpy.ndarray, shape (3,), dtype complex
        [d̃_γ(N), d̃_c(N), d̃_g(N)] for the given channel.
    """
    return np.array(
        [
            d_photon(N, channel, alpha_s_ref, alpha_em, mc, mu0_over_mc),
            d_charm(N, channel, alpha_s_ref, alpha_em, mc, mu0_over_mc),
            d_gluon(N, channel, alpha_s_ref, alpha_em, mc, mu0_over_mc),
        ],
        dtype=complex,
    )
