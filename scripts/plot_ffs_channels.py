#!/usr/bin/env python3
r"""Plot z · D_{i→J/ψ}(z, μ) broken down by NRQCD channel.

One PDF per parton component (γ, c, g); each PDF has four subplots,
one per NRQCD channel (³S₁^[1], ³S₁^[8], ¹S₀^[8], ³P_J^[8]).
Within each subplot, curves are coloured by evolution scale μ ∈ [3, 100] GeV.

Usage
-----
    python scripts/plot_ffs_channels.py              # saves to figs/
    python scripts/plot_ffs_channels.py -o out/      # saves to out/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import scienceplots  # noqa: F401 — registers the style sheets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from qedonia import evolve_ff_grid

plt.style.use(["science", "nature"])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LDMES: dict[str, float] = {
    "1S1": 1.16,  # ⟨O^{J/ψ}(³S₁^[1])⟩  [GeV³]  color-singlet
    "3S18": 1.06e-2,  # ⟨O^{J/ψ}(³S₁^[8])⟩  [GeV³]  color-octet
    "1S08": 8.9e-3,  # ⟨O^{J/ψ}(¹S₀^[8])⟩  [GeV³]  color-octet
    "3PJ8": 1.6e-2,  # ⟨O^{J/ψ}(³P_J^[8])⟩ [GeV⁵]  color-octet
}
MU_FROM: float = 3.0
ALPHA_S_REF: float = 0.26

MU_VALUES: list[float] = np.geomspace(3.0, 1e2, 8).tolist()
Z_VALUES: np.ndarray = np.geomspace(1e-4, 0.95, 200)

_CHANNELS: list[tuple[str, str]] = [
    ("1S1", r"${}^3\!S_1^{[1]}$"),
    ("3S18", r"${}^3\!S_1^{[8]}$"),
    ("1S08", r"${}^1\!S_0^{[8]}$"),
    ("3PJ8", r"${}^3\!P_J^{[8]}$"),
]

_COMPONENTS: list[tuple[int, str, str, str]] = [
    # (grid_index,  y-label, suptitle fragment, filestem)
    (0, r"$z\,D_{\gamma \to J/\!\psi}(z,\mu)$", r"$\gamma \to J/\!\psi$", "gamma"),
    (1, r"$z\,D_{c \to J/\!\psi}(z,\mu)$", r"$c \to J/\!\psi$", "c"),
    (2, r"$z\,D_{g \to J/\!\psi}(z,\mu)$", r"$g \to J/\!\psi$", "g"),
]

_MELLIN_KW = dict(
    n_steps=50,
    mellin_T=200,
    mellin_n_nodes=2001,
    lanczos=True,
    lanczos_order=2,
)


def compute_channel_grids() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run evolve_ff_grid once per channel; return z_arr, mu_arr, and per-channel
    grids.
    """
    channel_grids: dict[str, np.ndarray] = {}
    z_arr = mu_arr = np.empty(0)

    for channel, ldme in LDMES.items():
        z_arr, mu_arr, grid = evolve_ff_grid(
            ldmes={channel: ldme},
            mu_from=MU_FROM,
            mu_values=MU_VALUES,
            alpha_s_ref=ALPHA_S_REF,
            z_values=Z_VALUES,
            **_MELLIN_KW,
        )
        channel_grids[channel] = grid

    return z_arr, mu_arr, channel_grids


def make_figures(output_dir: str) -> None:
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Computing per-channel grids …")
    z_arr, mu_arr, channel_grids = compute_channel_grids()

    cmap = plt.get_cmap("plasma")
    norm = mcolors.LogNorm(vmin=mu_arr[0], vmax=mu_arr[-1])
    colors = [cmap(norm(mu)) for mu in mu_arr]

    for comp_idx, ylabel, comp_title, filestem in _COMPONENTS:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(6.8, 5.0),
            sharey=False,
            gridspec_kw={"wspace": 0.38, "hspace": 0.45},
        )

        for ax, (channel, ch_label) in zip(axes.flat, _CHANNELS):
            zD = z_arr[:, np.newaxis] * channel_grids[channel][comp_idx, :, :]

            for imu, color in enumerate(colors):
                ax.plot(z_arr, zD[:, imu], color=color, lw=0.6)

            ax.axhline(0, color="0.6", lw=0.5, ls="--", zorder=0)

            ax.set_xscale("log")
            ax.set_xlim(1e-4, 1)
            ax.set_xlabel(r"$z$")

            sci_fmt = ticker.ScalarFormatter(useMathText=True)
            sci_fmt.set_scientific(True)
            sci_fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(sci_fmt)

            ax.set_ylabel(ylabel, labelpad=3)
            ax.set_title(ch_label, pad=3)

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
            aspect=30,
        )
        cbar.set_label(r"$\mu\;[\mathrm{GeV}]$", labelpad=4)
        cbar.set_ticks(mu_arr.tolist())
        cbar.set_ticklabels([f"${m:.0f}$" if m >= 10 else f"${m:.1f}$" for m in mu_arr])
        cbar.minorticks_off()

        fig.suptitle(
            rf"$J/\psi$ FFs — {comp_title} by NRQCD channel"
            r" (LO QCD$\,\otimes\,$QED, $\mu_0 = 2m_c$)",
            y=1.01,
            fontsize=7,
        )

        path = out / f"jpsi_ffs_{filestem}_channels.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=450)
        plt.close(fig)
        print(f"Saved → {path.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output-dir",
        default="figs",
        help="Output directory (default: figs/)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_figures(args.output_dir)
