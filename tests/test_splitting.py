"""Verify all Mellin-space splitting functions against direct numerical quadrature.

Reproduces the verification scripts in Appendix A of notes_jpsi.tex.
All assertions use the tolerance thresholds from the notes.
"""

import pytest
from scipy.integrate import quad
from scipy.special import digamma

from qedonia.splitting import P_cc, P_cg, P_gc, P_gg, P_photon_c, P_c_photon, S1, beta0
from qedonia.constants import CF, CA, TF, EC2

_GAMMA_E = float(-digamma(1))
_N_VALUES = [2, 3, 4]


# ---------------------------------------------------------------------------
# Helper: S₁ reference
# ---------------------------------------------------------------------------


def _S1_ref(N: int) -> float:
    return sum(1.0 / k for k in range(1, N + 1))


@pytest.mark.parametrize("N", _N_VALUES)
def test_S1_vs_harmonic_sum(N):
    """S₁(N) from digamma must match the direct harmonic sum."""
    assert abs(S1(N) - _S1_ref(N)) < 1e-12


# ---------------------------------------------------------------------------
# Plus-distribution identity: (1/(1-z))_+  →  -S₁(N)  (eq. mellin_plus)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", _N_VALUES)
def test_plus_distribution_mellin(N):
    """∫₀¹ (z^{N-1}-1)/(1-z) dz  =  -S₁(N-1)  (standard z^{N-1} Mellin convention)."""
    val, _ = quad(
        lambda z: (z ** (N - 1) - 1.0) / (1.0 - z), 0.0, 1.0 - 1e-10, limit=200
    )
    assert abs(val - (-S1(N - 1))) < 1e-8


# ---------------------------------------------------------------------------
# Building block: z(1-z) → 1/[(N+1)(N+2)]  (eq. Mzz)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", _N_VALUES)
def test_zz_moment(N):
    val, _ = quad(lambda z: z ** (N - 1) * z * (1.0 - z), 0.0, 1.0)
    formula = 1.0 / ((N + 1) * (N + 2))
    assert abs(val - formula) < 1e-10


# ---------------------------------------------------------------------------
# Building block: photon shape (no-ln part) (eq. Mphotshape)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", _N_VALUES)
def test_photon_shape_no_ln(N):
    """Numerical integral of z(2-2z+z²)/(1-z) with plus-prescription (ln excluded)."""

    def integrand(z):
        poly = z ** (N - 1) * (z - z**2)  # from (z - z²) part
        plus = (z ** (N - 1) - 1.0) * z / (1.0 - z)  # from z * (1/(1-z))_+
        return poly + plus

    val, _ = quad(integrand, 0.0, 1.0 - 1e-10, limit=200)
    formula = 1.0 - float(S1(N).real) + 1.0 / (N + 1) - 1.0 / (N + 2)
    assert abs(val - formula) < 1e-8


# ---------------------------------------------------------------------------
# QCD splitting functions: P̃_qq, P̃_qg, P̃_gq, P̃_gg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", _N_VALUES)
def test_Pqq(N):
    """P̃_cc(N) = CF [3/2 - 2 S₁ + 1/(N(N+1))].

    ∫(z^{N-1}-1)(1+z²)/(1-z) dz already equals P̃_qq(N)/CF exactly because the
    3/2 δ(1-z) piece and the -3/2 correction from the plus-prescription cancel.
    """
    val, _ = quad(
        lambda z: CF * (z ** (N - 1) - 1.0) * (1.0 + z**2) / (1.0 - z),
        0.0,
        1.0 - 1e-10,
        limit=200,
    )
    assert abs(val - P_cc(N).real) < 1e-7


@pytest.mark.parametrize("N", _N_VALUES)
def test_Pqg(N):
    """P̃_cg(N) = CF [2/(N-1) - 2/N + 1/(N+1)]."""
    val, _ = quad(
        lambda z: z ** (N - 1) * CF * (1.0 + (1.0 - z) ** 2) / z,
        1e-14,
        1.0,
    )
    assert abs(val - P_cg(N).real) < 1e-7


@pytest.mark.parametrize("N", _N_VALUES)
def test_Pgq(N):
    """P̃_gc(N) = TF [1/N - 2/(N+1) + 2/(N+2)]  (single flavour)."""
    val, _ = quad(
        lambda z: z ** (N - 1) * TF * (z**2 + (1.0 - z) ** 2),
        0.0,
        1.0,
    )
    assert abs(val - P_gc(N).real) < 1e-10


@pytest.mark.parametrize("N", _N_VALUES)
def test_Pgg(N, nf=4):
    """P̃_gg(N) = 2CA[1-S₁+1/(N(N-1))+1/((N+1)(N+2))] + β₀/2."""
    b0 = beta0(nf)

    def integrand(z):
        if abs(z - 1) < 1e-12:
            return 0.0
        plus = 2.0 * CA * (z ** (N - 1) - 1.0) * z / (1.0 - z)
        rest = 2.0 * CA * z ** (N - 1) * ((1.0 - z) / z + z * (1.0 - z))
        return plus + rest

    val, _ = quad(integrand, 1e-14, 1.0 - 1e-10, limit=200)
    val += b0 / 2.0  # δ(1-z) piece gives β₀/2
    assert abs(val - P_gg(N, nf).real) < 1e-6


# ---------------------------------------------------------------------------
# QED splitting functions: P̃_γc, P̃_cγ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", _N_VALUES)
def test_P_photon_c(N):
    """P̃_γc(N) = ec² [1/N - 2/(N+1) + 2/(N+2)]."""
    val, _ = quad(
        lambda z: z ** (N - 1) * EC2 * (z**2 + (1.0 - z) ** 2),
        0.0,
        1.0,
    )
    assert abs(val - P_photon_c(N).real) < 1e-10


@pytest.mark.parametrize("N", _N_VALUES)
def test_P_c_photon(N):
    """P̃_cγ(N) = ec² [2/(N-1) - 2/N + 1/(N+1)]."""
    val, _ = quad(
        lambda z: z ** (N - 1) * EC2 * (1.0 + (1.0 - z) ** 2) / z,
        1e-14,
        1.0,
    )
    assert abs(val - P_c_photon(N).real) < 1e-8
