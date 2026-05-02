#!/usr/bin/env python3
r"""Braaten-Yuan initial-condition check and literature reference value.

At μ = μ₀ = 2mc the color-singlet (³S₁^[1]) charm FF is known analytically
via the Braaten-Yuan formula.  This script validates the pipeline at μ₀ by
comparing that formula against the Mellin-inverted output of evolve_ff_grid.

It also prints  z D_{c→J/ψ}(z=0.5, μ=91 GeV)  (full LDMES) as a reference
number for comparison with the literature.

Left panel  : z D_c^{1S1}(z, μ₀) — analytic (solid) vs Mellin-inverted (dashed)
Right panel : |relative difference| × 100 %

Usage
-----
    python scripts/plot_by_check.py
    python scripts/plot_by_check.py -o figs/by_check.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import scienceplots  # noqa: F401 — registers the style sheets
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from qedonia import evolve_ff_grid
from qedonia.constants import MC_GEV

plt.style.use(["science", "nature"])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MU_FROM: float = 3.0
ALPHA_S_REF: float = 0.26
MC: float = MC_GEV
LDME_1S1: float = 1.16

LDMES_FULL: dict[str, float] = {
    "1S1": 1.16,
    "3S18": 1.06e-2,
    "1S08": 8.9e-3,
    "3PJ8": 1.6e-2,
}

Z_ARR: np.ndarray = np.geomspace(0.01, 0.99, 300)


def _by_analytic(z: np.ndarray) -> np.ndarray:
    """z D_c^{1S1}(z, μ₀) directly from the Braaten-Yuan formula."""
    poly = 16.0 - 32.0 * z + 72.0 * z**2 - 32.0 * z**3 + 5.0 * z**4
    base = z * (1.0 - z) ** 2 / (2.0 - z) ** 6 * poly
    prefactor = LDME_1S1 * (16.0 * ALPHA_S_REF**2) / (729.0 * MC**3)
    return prefactor * z * base


def make_figure(output: str) -> None:
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Mellin-inverted IC at μ₀ (no Lanczos — BY IC is smooth)
    print("Computing Mellin-inverted IC at μ₀ …")
    z_arr, _, grid_ic = evolve_ff_grid(
        ldmes={"1S1": LDME_1S1},
        mu_from=MU_FROM,
        mu_values=[MU_FROM],
        alpha_s_ref=ALPHA_S_REF,
        z_values=Z_ARR,
        n_steps=1,
        mellin_T=200,
        mellin_n_nodes=2001,
        lanczos=False,
    )

    zD_mellin = z_arr * grid_ic[1, :, 0]
    zD_analytic = _by_analytic(z_arr)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(
            np.abs(zD_analytic) > 1e-50,
            np.abs(zD_mellin / zD_analytic - 1.0) * 100.0,
            np.nan,
        )

    # Literature comparison: z D_c(z=0.5, μ=91 GeV), full LDMES
    print("Computing z D_c(z=0.5, μ=91 GeV) for literature comparison …")
    _, _, grid_91 = evolve_ff_grid(
        ldmes=LDMES_FULL,
        mu_from=MU_FROM,
        mu_values=[91.0],
        alpha_s_ref=ALPHA_S_REF,
        z_values=np.array([0.5]),
        n_steps=50,
        mellin_T=200,
        mellin_n_nodes=2001,
        lanczos=True,
        lanczos_order=2,
    )
    zD_c_91 = float(0.5 * grid_91[1, 0, 0])
    print(f"  z D_{{c→J/ψ}}(z=0.50, μ=91 GeV) = {zD_c_91:.4e}  (full LDMES, Fejer)")

    # Integrated rate ∫₀¹ dz D_c(z, M_Z) for the color-singlet channel
    print("Computing ∫D_c dz at μ=91 GeV (1S1 channel) …")
    z_dense = np.linspace(0.0, 1.0, 4001)[1:-1]
    _, _, grid_int = evolve_ff_grid(
        ldmes={"1S1": LDME_1S1},
        mu_from=MU_FROM,
        mu_values=[MU_FROM, 91.0],
        alpha_s_ref=ALPHA_S_REF,
        z_values=z_dense,
        n_steps=50,
        mellin_T=200,
        mellin_n_nodes=2001,
        lanczos=True,
        lanczos_order=2,
    )
    int_Dc_mu0 = float(np.trapezoid(grid_int[1, :, 0], z_dense))
    int_Dc_91 = float(np.trapezoid(grid_int[1, :, 1], z_dense))
    print(f"  ∫₀¹ D_c dz  at μ₀ = {MU_FROM} GeV  = {int_Dc_mu0:.4e}")
    print(f"  ∫₀¹ D_c dz  at μ  = 91  GeV  = {int_Dc_91:.4e}")

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.6),
        gridspec_kw={"wspace": 0.40},
    )

    # Left: BY formula vs Mellin inversion
    ax_l.plot(z_arr, zD_analytic, color="#1f77b4", lw=0.9, label="analytic (BY)")
    ax_l.plot(z_arr, zD_mellin, color="#ff7f0e", lw=0.9, ls="--", label="Mellin-inv.")

    ax_l.set_xscale("log")
    ax_l.set_xlim(z_arr[0], z_arr[-1])
    ax_l.set_xlabel(r"$z$")

    sci = ticker.ScalarFormatter(useMathText=True)
    sci.set_scientific(True)
    sci.set_powerlimits((0, 0))
    ax_l.yaxis.set_major_formatter(sci)

    ax_l.set_ylabel(r"$z\,D_{c \to J/\!\psi}^{{}^3\!S_1^{[1]}}(z,\mu_0)$", labelpad=3)
    ax_l.legend(fontsize=5, handlelength=1.2, frameon=False)

    ax_r.plot(z_arr, rel_err, color="#2ca02c", lw=0.9)
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlim(z_arr[0], z_arr[-1])
    ax_r.set_xlabel(r"$z$")
    ax_r.set_ylabel(r"rel.\ error [\%]", labelpad=3)
    ax_r.set_title(r"Mellin-inv.\ vs analytic", pad=3)

    fig.suptitle(
        r"Braaten-Yuan IC check — ${}^3\!S_1^{[1]}$ charm FF at $\mu_0=2m_c$",
        y=1.02,
        fontsize=7,
    )

    fig.savefig(out, bbox_inches="tight", dpi=450)
    print(f"Saved → {out.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        default="figs/by_check.pdf",
        help="Output path (default: figs/by_check.pdf)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_figure(args.output)
