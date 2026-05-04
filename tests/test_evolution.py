"""Smoke-tests and sanity checks for the evolution machinery."""

import numpy as np
import pytest

from qedonia.evolution import transfer_matrix, evolve_vector
from qedonia.constants import MC_GEV, MU0_OVER_MC

_MC = MC_GEV
_MU0 = MU0_OVER_MC * _MC
_MU2_0 = _MU0**2
_AS_REF = 0.26
_N = 2.0 + 0j
_NF = 4

_ALPHA_EM = 1.0 / 133.0
_MU2_MZ = 91.0**2


def test_transfer_matrix_identity_at_zero_evolution():
    """E(N; μ² → μ²) must be the identity."""
    E = transfer_matrix(_N, _MU2_0, _MU2_0, _AS_REF, _MU2_0, nf=_NF, n_steps=1)
    assert np.allclose(E, np.eye(3), atol=1e-12)


def test_transfer_matrix_shape():
    E = transfer_matrix(_N, _MU2_0, 100.0, _AS_REF, _MU2_0, nf=_NF, n_steps=20)
    assert E.shape == (3, 3)
    assert E.dtype == complex


def test_gluon_photon_decoupled():
    """Γ_γg = Γ_gγ = 0 at LO — there is no direct photon↔gluon coupling.
    (Indirect γ←c←g mixing is allowed and expected.)"""
    from qedonia.splitting import gamma_matrix

    Gamma = gamma_matrix(_N, as_over_2pi=0.05, aem_over_2pi=1e-4, nf=_NF)
    assert abs(Gamma[0, 2]) < 1e-30, "Direct γ←g coupling must vanish"
    assert abs(Gamma[2, 0]) < 1e-30, "Direct g←γ coupling must vanish"


def test_charm_sources_photon():
    """A non-zero charm IC must generate a photon component via QED mixing."""
    D0 = np.array([0.0, 1.0, 0.0], dtype=complex)
    D1 = evolve_vector(
        D0,
        _N,
        _MU2_0,
        100.0,
        _AS_REF,
        _MU2_0,
        nf=_NF,
        n_steps=50,
    )
    assert abs(D1[0]) > 0.0, "Charm → photon QED mixing must produce D_γ > 0"


@pytest.mark.parametrize("n_steps", [10, 50, 200])
def test_convergence_with_steps(n_steps):
    """Transfer matrix must converge as n_steps increases."""
    E_ref = transfer_matrix(
        _N,
        _MU2_0,
        100.0,
        _AS_REF,
        _MU2_0,
        nf=_NF,
        n_steps=200,
    )
    E_test = transfer_matrix(
        _N,
        _MU2_0,
        100.0,
        _AS_REF,
        _MU2_0,
        nf=_NF,
        n_steps=n_steps,
    )
    # The (c,c) element carries most of the weight
    tol = 5e-2 if n_steps == 10 else 1e-3
    assert abs(E_test[1, 1] - E_ref[1, 1]) / abs(E_ref[1, 1]) < tol


def test_running_aem_identity_at_initial_scale():
    """transfer_matrix with run_alpha_em=True must equal fixed at μ_from itself."""
    E_fixed = transfer_matrix(
        _N,
        _MU2_0,
        _MU2_0,
        _AS_REF,
        _MU2_0,
        alpha_em=_ALPHA_EM,
        nf=_NF,
        n_steps=1,
    )
    E_run = transfer_matrix(
        _N,
        _MU2_0,
        _MU2_0,
        _AS_REF,
        _MU2_0,
        alpha_em=_ALPHA_EM,
        nf=_NF,
        n_steps=1,
        run_alpha_em=True,
    )
    assert np.allclose(E_fixed, E_run, atol=1e-12)


def test_running_aem_differs_from_fixed():
    """Running α produces a slightly different photon component than fixed α."""
    D0 = np.array([0.0, 1.0, 0.0], dtype=complex)
    D_fixed = evolve_vector(
        D0,
        _N,
        _MU2_0,
        _MU2_MZ,
        _AS_REF,
        _MU2_0,
        alpha_em=_ALPHA_EM,
        nf=_NF,
        n_steps=50,
        run_alpha_em=False,
    )
    D_run = evolve_vector(
        D0,
        _N,
        _MU2_0,
        _MU2_MZ,
        _AS_REF,
        _MU2_0,
        alpha_em=_ALPHA_EM,
        nf=_NF,
        n_steps=50,
        run_alpha_em=True,
    )
    assert abs(D_run[0]) != abs(
        D_fixed[0]
    ), "D_γ must differ between fixed and running α"
    rel_diff_charm = abs(D_run[1] - D_fixed[1]) / abs(D_fixed[1])
    assert rel_diff_charm < 0.01, "Running α must change D_c by < 1%"


def test_running_aem_photon_larger():
    """Running α is larger than fixed α at M_Z, so D_γ must be larger with running."""
    D0 = np.array([0.0, 1.0, 0.0], dtype=complex)
    D_fixed = evolve_vector(
        D0,
        _N,
        _MU2_0,
        _MU2_MZ,
        _AS_REF,
        _MU2_0,
        alpha_em=_ALPHA_EM,
        nf=_NF,
        n_steps=50,
        run_alpha_em=False,
    )
    D_run = evolve_vector(
        D0,
        _N,
        _MU2_0,
        _MU2_MZ,
        _AS_REF,
        _MU2_0,
        alpha_em=_ALPHA_EM,
        nf=_NF,
        n_steps=50,
        run_alpha_em=True,
    )
    # The off-diagonal generation term dominates over the diagonal decay:
    # larger α(μ) at high scales → more photon production → D_γ larger
    assert abs(D_run[0]) > abs(D_fixed[0])
