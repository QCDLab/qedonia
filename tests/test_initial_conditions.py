"""Verify NRQCD initial conditions against direct numerical quadrature.

Reproduces the verification scripts in Appendix A.3 of notes_jpsi.tex.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import digamma

from qedonia.initial_conditions import (
    F_gamma,
    _zz_moment,
    braaten_yuan_moment,
    d_charm,
    d_gluon,
)

_GAMMA_E = float(-digamma(1))
_N_VALUES = [2, 3, 4]

_MC = 1.5
_MU0_OVER_MC = 2.0
_AS_REF = 0.26
_AEM = 1.0 / 133.0

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", _N_VALUES)
def test_zz_moment(N):
    """∫₀¹ z^{N-1} z(1-z) dz = 1/[(N+1)(N+2)]."""
    val, _ = quad(lambda z: z ** (N - 1) * z * (1.0 - z), 0.0, 1.0)
    assert abs(val - _zz_moment(N).real) < 1e-10


@pytest.mark.parametrize("N", _N_VALUES)
def test_F_gamma(N):
    """Photon-shape Mellin moment F_γ(N) including the ln(μ₀²/mc²) term."""
    ln_ratio = np.log(_MU0_OVER_MC**2)

    def integrand(z):
        poly = z ** (N - 1) * (z - z**2)  # from (z - z²)
        plus = (z ** (N - 1) - 1.0) * z / (1.0 - z)  # from z*(1/(1-z))_+
        return poly + plus

    val, _ = quad(integrand, 0.0, 1.0 - 1e-10, limit=200)
    val += ln_ratio  # δ(1-z) piece from plus-prescription
    assert abs(val - F_gamma(N, _MU0_OVER_MC).real) < 1e-8


# ---------------------------------------------------------------------------
# Braaten–Yuan moment B(N)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", _N_VALUES)
def test_braaten_yuan_real(N):
    """B(N) via quad must match braaten_yuan_moment for integer N."""

    def integrand(z):
        poly = 16.0 - 32.0 * z + 72.0 * z**2 - 32.0 * z**3 + 5.0 * z**4
        return z ** (N - 1) * z * (1.0 - z) ** 2 / (2.0 - z) ** 6 * poly

    ref, _ = quad(integrand, 0.0, 1.0, limit=200)
    assert abs(braaten_yuan_moment(complex(N)).real - ref) < 1e-9


# ---------------------------------------------------------------------------
# Color-factor ratio: g/c for ³S₁^[8] = 1/4
# ---------------------------------------------------------------------------


def test_color_ratio_3S18():
    """d̃_g^{3S18} / d̃_c^{3S18} = (π αs/24) / (π αs/6) = 1/4."""
    ratio = (
        d_gluon(2.0 + 0j, "3S18", _AS_REF, _AEM, _MC, _MU0_OVER_MC).real
        / d_charm(2.0 + 0j, "3S18", _AS_REF, _AEM, _MC, _MU0_OVER_MC).real
    )
    assert abs(ratio - 0.25) < 1e-14


# ---------------------------------------------------------------------------
# Delta(1-z) channels: d̃^{3S18} is constant in N
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N1,N2", [(2, 3), (2, 4), (3, 5)])
def test_3S18_constant_in_N_charm(N1, N2):
    v1 = d_charm(complex(N1), "3S18", _AS_REF, _AEM, _MC, _MU0_OVER_MC)
    v2 = d_charm(complex(N2), "3S18", _AS_REF, _AEM, _MC, _MU0_OVER_MC)
    assert abs(v1 - v2) < 1e-14, "d̃_c^{3S18} must be constant in N"


@pytest.mark.parametrize("N1,N2", [(2, 3), (2, 4), (3, 5)])
def test_3S18_constant_in_N_gluon(N1, N2):
    v1 = d_gluon(complex(N1), "3S18", _AS_REF, _AEM, _MC, _MU0_OVER_MC)
    v2 = d_gluon(complex(N2), "3S18", _AS_REF, _AEM, _MC, _MU0_OVER_MC)
    assert abs(v1 - v2) < 1e-14, "d̃_g^{3S18} must be constant in N"


# ---------------------------------------------------------------------------
# Vanishing gluon color-singlet IC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", _N_VALUES)
def test_gluon_1S1_vanishes(N):
    val = d_gluon(complex(N), "1S1", _AS_REF, _AEM, _MC, _MU0_OVER_MC)
    assert abs(val) < 1e-30
