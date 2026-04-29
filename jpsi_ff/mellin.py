r"""Mellin contour and numerical inverse Mellin transform.

Convention
----------
Forward transform:   D̃(N) = ∫₀¹ z^{N-1} D(z) dz
Inverse transform:   D(z) = (1/2πi) ∫_{c-i∞}^{c+i∞} z^{-N} D̃(N) dN

On the straight-line contour N = c + it,  dN = i dt:

    D(z) = (1/2π) ∫_{-T}^{T} D̃(c+it) e^{-(c+it) ln z} dt

which is real for physical FFs.  We integrate on the right half
  D(z) = (1/π) ∫₀^T Re[ D̃(c+it) e^{-(c+it) ln z} ] dt
and use the symmetry D̃(c−it) = D̃(c+it)* (when D(z) is real).

Parameter guidance
------------------
* c = 2.0   — safely right of the rightmost pole at N = 1
              (from P̃_cg and P̃_gg which diverge as 1/(N−1)).
* T = 60.0  — captures the rapid decay of D̃(N) for smooth FFs.
              May need increasing for very small z (z < 0.01).
* n_nodes   — trapezoidal rule on [0, T] with n_nodes points.
              n_nodes = 601 gives < 0.1 % accuracy for z ≥ 0.05.
"""

from __future__ import annotations

import numpy as np

# Default contour parameters
DEFAULT_C: float = 2.0
DEFAULT_T: float = 60.0
DEFAULT_N_NODES: int = 601


def contour_nodes(
    c: float = DEFAULT_C,
    T: float = DEFAULT_T,
    n_nodes: int = DEFAULT_N_NODES,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return (N_values, weights) for the quadrature on the Mellin contour.

    The nodes are  N_k = c + i t_k  with  t_k ∈ (0, T].
    Weights implement the trapezoidal rule for  (1/π) ∫₀^T · dt.

    Returns
    -------
    N_values : ndarray, shape (n_nodes,), complex
    weights  : ndarray, shape (n_nodes,), float
    """
    t_vals = np.linspace(0.0, T, n_nodes)
    dt = T / (n_nodes - 1)
    weights = np.full(n_nodes, dt / np.pi)
    weights[0] *= 0.5   # trapezoidal endpoint correction
    weights[-1] *= 0.5
    return (c + 1j * t_vals).astype(complex), weights


def invert_mellin(
    D_tilde: np.ndarray,
    N_values: np.ndarray,
    weights: np.ndarray,
    z_values: np.ndarray,
) -> np.ndarray:
    r"""Evaluate D(z) at each z by quadrature from precomputed D̃ values.

        D(z_k) ≈ (1/π) Σ_j  Re[ D̃(N_j) · e^{-N_j ln z_k} ] · w_j

    Parameters
    ----------
    D_tilde : ndarray, shape (n_nodes,), complex
        D̃(N_j) evaluated at the contour nodes.
    N_values : ndarray, shape (n_nodes,), complex
        Contour nodes (from :func:`contour_nodes`).
    weights : ndarray, shape (n_nodes,), float
        Quadrature weights (from :func:`contour_nodes`).
    z_values : ndarray, shape (n_z,), float
        Points in (0, 1) at which to return D(z).

    Returns
    -------
    ndarray, shape (n_z,), float
        Real-valued D(z).
    """
    z_arr = np.asarray(z_values, dtype=float)
    result = np.empty(len(z_arr))
    for k, z in enumerate(z_arr):
        # e^{-N ln z} = z^{-N}
        integrand = D_tilde * np.exp(-N_values * np.log(z))
        result[k] = float(np.dot(np.real(integrand), weights))
    return result
