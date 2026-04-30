#!/usr/bin/env python3
r"""Plot z · D_{i→J/ψ}(z, μ) for i ∈ {γ, c, g} at eight evolution scales.

Three subplots share the same z-axis; a shared logarithmic colorbar encodes
the evolution scale from μ = 3 GeV to μ = 100 GeV.

Usage
-----
    python scripts/plot_ffs.py                       # saves figs/jpsi_ffs.pdf
    python scripts/plot_ffs.py -o my_plot.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from qedonia import evolve_ff_grid
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use(["science", "nature"])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LDMES: dict[str, float] = {
    "1S1": 1.16,  # ⟨O^{J/ψ}(³S₁^[1])⟩  [GeV³]  color-singlet
    "3S18": 1.06e-2,  # ⟨O^{J/ψ}(³S₁^[8])⟩  [GeV³]  color-octet
    "1S08": 8.9e-3,  # ⟨O^{J/ψ}(¹S₀^[8])⟩  [GeV³]  color-octet
    "3PJ8": 1.6e-2,  # ⟨O^{J/ψ}(³P_J^[8])⟩ [GeV⁵]  color-octet
}
MU_FROM: float = 3.0  # μ₀  [GeV]
ALPHA_S_REF: float = 0.26  # αs(μ₀)

# Eight log-spaced target scales μ₀ → 10² GeV
MU_VALUES: list[float] = np.geomspace(3.0, 1e2, 8).tolist()

# z grid: full kinematic range [1e-4, 0.95]; upper limit avoids endpoint artifacts
Z_VALUES: np.ndarray = np.geomspace(1e-4, 0.95, 200)

_PANELS = [
    # (grid_index,  y-label,                        panel_title)
    (0, r"$z\,D_{\gamma \to J/\!\psi}(z,\mu)$", r"$\gamma \to J/\!\psi$"),
    (1, r"$z\,D_{c \to J/\!\psi}(z,\mu)$", r"$c \to J/\!\psi$"),
    (2, r"$z\,D_{g \to J/\!\psi}(z,\mu)$", r"$g \to J/\!\psi$"),
]


def make_figure(output: str) -> None:
    # Compute all components × all scales in a single call
    z_arr, mu_arr, grid = evolve_ff_grid(
        ldmes=LDMES,
        mu_from=MU_FROM,
        mu_values=MU_VALUES,
        alpha_s_ref=ALPHA_S_REF,
        z_values=Z_VALUES,
        n_steps=50,
        mellin_T=200,
        mellin_n_nodes=2001,
        lanczos=True,
        lanczos_order=2,
    )

    # Logarithmic colormap: light (low μ) → dark (high μ)
    cmap = plt.get_cmap("plasma")
    norm = mcolors.LogNorm(vmin=mu_arr[0], vmax=mu_arr[-1])
    colors = [cmap(norm(mu)) for mu in mu_arr]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.8, 2.4),
        sharey=False,
        gridspec_kw={"wspace": 0.38},
    )

    for ax, (comp_idx, ylabel, title) in zip(axes, _PANELS):
        zD = z_arr[:, np.newaxis] * grid[comp_idx, :, :]  # (n_z, n_mu)

        for imu, color in enumerate(colors):
            ax.plot(z_arr, zD[:, imu], color=color, lw=0.9)

        ax.axhline(0, color="0.6", lw=0.5, ls="--", zorder=0)

        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_xlabel(r"$z$")

        sci_fmt = ticker.ScalarFormatter(useMathText=True)
        sci_fmt.set_scientific(True)
        sci_fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(sci_fmt)

        ax.set_ylabel(ylabel, labelpad=3)
        ax.set_title(title, pad=3)

        ymax = np.nanmax(np.abs(zD)) * 1.15
        ax.set_ylim(-0.05 * ymax, ymax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="vertical",
        fraction=0.018,
        pad=0.015,
        aspect=25,
    )
    cbar.set_label(r"$\mu\;[\mathrm{GeV}]$", labelpad=4)
    cbar.set_ticks(mu_arr)
    cbar.set_ticklabels([f"${m:.0f}$" if m >= 10 else f"${m:.1f}$" for m in mu_arr])
    cbar.minorticks_off()

    fig.suptitle(
        r"$J/\psi$ FFs — LO QCD$\,\otimes\,$QED DGLAP ($\mu_0 = 2m_c$)",
        y=1.02,
        fontsize=7,
    )

    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        default="figs/jpsi_ffs.pdf",
        help="Output path (default: figs/jpsi_ffs.pdf)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_figure(args.output)
