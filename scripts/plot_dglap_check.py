#!/usr/bin/env python3
r"""DGLAP derivative consistency check.

For each Mellin moment N and parton component (γ, c, g), compare the
right-hand side of the DGLAP equation evaluated two independent ways:

  • Γ(N, αs(μ)) · D̃(N, μ)              — direct from gamma_matrix
  • [D̃(N, μ·(1+ε)) − D̃(N, μ·(1−ε))] / Δ(ln μ²)   — finite difference

Top row    : |Γ · D̃(N, μ)|  vs  μ   (log-log; solid = Γ·D̃, dashed = FD)
Bottom row : |FD / (Γ·D̃) − 1| × 100 %  (relative residual)

Usage
-----
    python scripts/plot_dglap_check.py
    python scripts/plot_dglap_check.py -o figs/dglap_check.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import scienceplots  # noqa: F401 — registers the style sheets
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from qedonia.constants import ALPHA_EM, MC_GEV, MU0_OVER_MC, NF
from qedonia.couplings import as_over_2pi
from qedonia.evolution import transfer_matrix
from qedonia.initial_conditions import CHANNELS, initial_vector
from qedonia.splitting import gamma_matrix

plt.style.use(["science", "nature"])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LDMES: dict[str, float] = {
    "1S1": 1.16,
    "3S18": 1.06e-2,
    "1S08": 8.9e-3,
    "3PJ8": 1.6e-2,
}
MU_FROM: float = 3.0
ALPHA_S_REF: float = 0.26
EPS: float = 1e-2  # fractional step in μ for the finite difference

MU_ARR: np.ndarray = np.geomspace(3.0, 1e2, 25)
N_TEST: list[int] = [3, 5, 8]

_COMPONENTS: list[tuple[int, str, str]] = [
    (0, r"$|\Gamma\tilde{D}|_{\gamma}$", r"$\gamma \to J/\!\psi$"),
    (1, r"$|\Gamma\tilde{D}|_{c}$", r"$c \to J/\!\psi$"),
    (2, r"$|\Gamma\tilde{D}|_{g}$", r"$g \to J/\!\psi$"),
]

_N_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def _build_d0(N: complex) -> np.ndarray:
    D0 = np.zeros(3, dtype=complex)
    for ch in CHANNELS:
        ldme = LDMES.get(ch, 0.0)
        if ldme:
            D0 += ldme * initial_vector(
                N,
                ch,
                ALPHA_S_REF,
                ALPHA_EM,
                MC_GEV,
                MU0_OVER_MC,
            )
    return D0


def _compute(N_values: list[int], mu_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return gamma_dot[iN, comp, imu] and fd[iN, comp, imu]."""
    mu2_from = MU_FROM**2
    mu2_ref = (MU0_OVER_MC * MC_GEV) ** 2
    aem_2pi = ALPHA_EM / (2.0 * np.pi)

    n_N = len(N_values)
    n_mu = len(mu_arr)
    gamma_dot = np.empty((n_N, 3, n_mu))
    fd_arr = np.empty((n_N, 3, n_mu))

    for iN, N in enumerate(N_values):
        Nc = complex(N)
        D0 = _build_d0(Nc)

        for imu, mu in enumerate(mu_arr):
            mu2 = mu**2
            mu2_p = (mu * (1.0 + EPS)) ** 2
            mu2_m = (mu * (1.0 - EPS)) ** 2
            dln = np.log(mu2_p) - np.log(mu2_m)

            D_mid = np.real(
                transfer_matrix(
                    Nc,
                    mu2_from,
                    mu2,
                    ALPHA_S_REF,
                    mu2_ref,
                    ALPHA_EM,
                    NF,
                    50,
                )
                @ D0
            )
            D_plus = np.real(
                transfer_matrix(
                    Nc,
                    mu2_from,
                    mu2_p,
                    ALPHA_S_REF,
                    mu2_ref,
                    ALPHA_EM,
                    NF,
                    50,
                )
                @ D0
            )
            D_minus = np.real(
                transfer_matrix(
                    Nc,
                    mu2_from,
                    mu2_m,
                    ALPHA_S_REF,
                    mu2_ref,
                    ALPHA_EM,
                    NF,
                    50,
                )
                @ D0
            )
            fd_arr[iN, :, imu] = (D_plus - D_minus) / dln

            a_s_2pi = float(as_over_2pi(ALPHA_S_REF, mu2_ref, mu2, NF))
            Gam = gamma_matrix(Nc, a_s_2pi, aem_2pi, NF)
            gamma_dot[iN, :, imu] = np.real(Gam @ D_mid.astype(complex))

    return gamma_dot, fd_arr


def make_figure(output: str) -> None:
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Computing DGLAP derivatives …")
    gamma_dot, fd_arr = _compute(N_TEST, MU_ARR)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(6.8, 4.4),
        gridspec_kw={"hspace": 0.52, "wspace": 0.44},
    )

    for ic, (comp_idx, ylabel_top, comp_title) in enumerate(_COMPONENTS):
        ax_top = axes[0, ic]
        ax_bot = axes[1, ic]

        for iN, N in enumerate(N_TEST):
            color = _N_COLORS[iN]
            gd = np.abs(gamma_dot[iN, comp_idx, :])
            fd = np.abs(fd_arr[iN, comp_idx, :])

            ax_top.plot(MU_ARR, gd, color=color, lw=0.8, label=f"$N={N}$")
            ax_top.plot(MU_ARR, fd, color=color, lw=0.8, ls="--", alpha=0.7)

            with np.errstate(divide="ignore", invalid="ignore"):
                denom = gamma_dot[iN, comp_idx, :]
                rel = np.where(
                    np.abs(denom) > 1e-50,
                    np.abs(fd_arr[iN, comp_idx, :] / denom - 1.0) * 100.0,
                    np.nan,
                )
            ax_bot.plot(MU_ARR, rel, color=color, lw=0.8)

        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.set_xlim(MU_ARR[0], MU_ARR[-1])
        ax_top.set_xlabel(r"$\mu\;[\mathrm{GeV}]$")
        ax_top.set_ylabel(ylabel_top, labelpad=3)
        ax_top.set_title(comp_title, pad=3)

        ax_bot.set_xscale("log")
        ax_bot.set_yscale("log")
        ax_bot.set_xlim(MU_ARR[0], MU_ARR[-1])
        ax_bot.set_xlabel(r"$\mu\;[\mathrm{GeV}]$")
        ax_bot.set_ylabel(r"$|\mathrm{FD}/(\Gamma\tilde{D})-1|\;[\%]$", labelpad=3)

    axes[0, 0].legend(fontsize=5, handlelength=1.0, frameon=False)
    axes[0, 2].legend(
        handles=[
            mlines.Line2D(
                [], [], color="0.35", lw=0.8, label=r"$\Gamma\cdot\tilde{D}$"
            ),
            mlines.Line2D([], [], color="0.35", lw=0.8, ls="--", label="FD"),
        ],
        fontsize=5,
        handlelength=1.2,
        frameon=False,
    )

    fig.suptitle(
        r"DGLAP derivative check — $J/\psi$ FFs"
        r" (LO QCD$\,\otimes\,$QED, $\mu_0=2m_c$, $\varepsilon=" + f"{EPS:.0e}" + r"$)",
        y=1.01,
        fontsize=7,
    )

    fig.savefig(out, bbox_inches="tight", dpi=450)
    print(f"Saved → {out.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        default="figs/dglap_check.pdf",
        help="Output path (default: figs/dglap_check.pdf)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_figure(args.output)
