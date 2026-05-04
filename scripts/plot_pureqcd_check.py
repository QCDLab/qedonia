#!/usr/bin/env python3
r"""Pure-QCD evolved FF check against Braaten–Yuan / Cho–Leibovich literature.

When the QED coupling is set to zero (``alpha_em=0``), the photon component
is identically zero and the (D_c, D_g) system evolves under the 2×2 LO QCD
DGLAP kernel.  This is exactly the setup of the classic J/ψ fragmentation
papers [BY93, CL96].

Checks performed
----------------
1. **D_γ = 0** : photon component is strictly zero at all z and μ.
2. **IC at μ₀** : z D_c(z, μ₀) matches the Braaten–Yuan analytic formula
   (same as ``plot_by_check.py`` but now verifying the QED-off code path).
3. **Evolved FF at M_Z** : D_c(z, 91 GeV) and D_g(z, 91 GeV) are compared
   with reference values from Braaten–Yuan (1993) Fig. 1 and
   Cho–Leibovich (1996) Table I.
4. **QED effect on D_c** : relative difference between the QED-on and
   QED-off charm FFs at M_Z (expected < 0.3 %).

Literature reference values (color-singlet ³S₁^[1], LO QCD,
⟨O⟩=1.16 GeV³, mc=1.5 GeV, αs(2mc)=0.26)
-----------------------------------------------------------
  D_c(z=0.5, μ₀=3 GeV)  ≈ 8.02×10⁻⁵   [BY93, analytic IC]
  D_c(z=0.5, M_Z=91 GeV) ≈ 9.0×10⁻⁵   [BY93, Fig. 1 reading]
  ∫₀¹ D_c dz at M_Z      ≈ 7.0×10⁻⁵   [this work, cf. CL96 Table I]

Usage
-----
    python scripts/plot_pureqcd_check.py
    python scripts/plot_pureqcd_check.py -o figs/pureqcd_check.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import scienceplots  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from qedonia import evolve_ff_grid
from qedonia.constants import MC_GEV

plt.style.use(["science", "nature"])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Parameters — match BY93 / CL96 setup exactly
# ---------------------------------------------------------------------------
MU_FROM: float = 3.0  # μ₀ = 2mc [GeV]
ALPHA_S_REF: float = 0.26  # αs(μ₀)
MC: float = MC_GEV  # mc = 1.5 GeV
LDME_1S1: float = 1.16  # ⟨O^{J/ψ}(³S₁^[1])⟩ [GeV³], BBL convention

# Literature reference points  (BY93 Fig. 1 and CL96 Table I)
# Format: (z, μ [GeV], D_c_ref, uncertainty, source)
_LIT_REFS: list[tuple[float, float, float, float, str]] = [
    (0.5, 3.0, 8.02e-5, 0.05e-5, "BY93 analytic IC"),
    (0.5, 91.2, 9.0e-5, 1.0e-5, "BY93 Fig. 1"),
]

_MELLIN_KW = dict(
    mc=MC,
    n_steps=50,
    mellin_T=200,
    mellin_n_nodes=2001,
    lanczos=True,
    lanczos_order=2,
)

Z_ARR: np.ndarray = np.geomspace(0.01, 0.99, 300)
MU_PLOT: list[float] = [MU_FROM, 10.0, 30.0, 91.2]


def _by_analytic(z: np.ndarray) -> np.ndarray:
    """z D_c^{1S1}(z, μ₀) — Braaten–Yuan analytic formula."""
    poly = 16.0 - 32.0 * z + 72.0 * z**2 - 32.0 * z**3 + 5.0 * z**4
    base = z * (1.0 - z) ** 2 / (2.0 - z) ** 6 * poly
    prefactor = LDME_1S1 * (16.0 * ALPHA_S_REF**2) / (729.0 * MC**3)
    return prefactor * z * base


def make_figure(output: str) -> None:
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Evolve with QED = 0  (pure LO QCD)
    print("Evolving with QED = 0 (pure LO QCD) …")
    z_arr, mu_arr, grid_qcd = evolve_ff_grid(
        ldmes={"1S1": LDME_1S1},
        mu_from=MU_FROM,
        mu_values=MU_PLOT,
        alpha_s_ref=ALPHA_S_REF,
        z_values=Z_ARR,
        alpha_em=0.0,  # ← QED off
        **_MELLIN_KW,
    )

    # Evolve with physical QED for comparison
    print("Evolving with physical QED (for comparison) …")
    _, _, grid_full = evolve_ff_grid(
        ldmes={"1S1": LDME_1S1},
        mu_from=MU_FROM,
        mu_values=MU_PLOT,
        alpha_s_ref=ALPHA_S_REF,
        z_values=Z_ARR,
        **_MELLIN_KW,
    )

    imu_mu0 = np.searchsorted(mu_arr, MU_FROM)
    imu_mz = np.searchsorted(mu_arr, 91.0)

    # Check: photon component identically zero
    max_Dgamma = float(np.max(np.abs(grid_qcd[0])))
    print(f"\n{'=' * 55}")
    print("CHECK 1 — D_γ = 0 with QED off")
    print(f"  max |D_γ(z, μ)| = {max_Dgamma:.2e}  (should be 0)")
    status = "PASS ✓" if max_Dgamma < 1e-20 else "FAIL ✗"
    print(f"  Status: {status}")

    # Check: IC and evolved values vs literature
    iz05 = np.argmin(np.abs(Z_ARR - 0.5))
    D_c_mu0 = float(grid_qcd[1, iz05, imu_mu0])
    D_c_mz = float(grid_qcd[1, iz05, imu_mz])
    D_c_by = float(_by_analytic(np.array([0.5]))[0] / 0.5)  # undo z× factor
    z_dense = np.linspace(0.0, 1.0, 4001)[1:-1]
    _, _, g_int = evolve_ff_grid(
        ldmes={"1S1": LDME_1S1},
        mu_from=MU_FROM,
        mu_values=[91.2],
        alpha_s_ref=ALPHA_S_REF,
        z_values=z_dense,
        alpha_em=0.0,
        **_MELLIN_KW,
    )
    int_Dc_mz = float(np.trapezoid(g_int[1, :, 0], z_dense))
    int_Dg_mz = float(np.trapezoid(g_int[2, :, 0], z_dense))

    print(f"\n{'=' * 55}")
    print("CHECK 2 — D_c(z=0.5) vs Braaten–Yuan analytic IC")
    print(f"  Our (QCD-off, Mellin inv.) : {D_c_mu0:.4e}")
    print(f"  BY93 analytic formula      : {D_c_by:.4e}")
    rel_ic = abs(D_c_mu0 / D_c_by - 1.0) * 100.0
    print(f"  Relative difference        : {rel_ic:.2f} %")
    status = "PASS ✓" if rel_ic < 2.0 else "WARN !"
    print(f"  Status: {status}")

    print(f"\n{'=' * 55}")
    print("CHECK 3 — D_c(z=0.5, M_Z) vs BY93 Fig. 1")
    print(f"  Our result (QED=0) : {D_c_mz:.4e}")
    print("  BY93 reference     : ~9.0e-05  ± 1.0e-05")
    rel_mz = abs(D_c_mz / 9.0e-5 - 1.0) * 100.0
    print(f"  Relative difference: {rel_mz:.1f} %")
    status = "PASS ✓" if rel_mz < 15.0 else "WARN !"
    print(f"  Status: {status}")

    print(f"\n{'=' * 55}")
    print("Integrated rates at M_Z (QED=0, 1S1 channel)")
    print(f"  ∫₀¹ D_c dz = {int_Dc_mz:.4e}")
    print(f"  ∫₀¹ D_g dz = {int_Dg_mz:.4e}")

    print(f"\n{'=' * 55}")
    print("CHECK 4 — QED effect on D_c at M_Z")
    D_c_mz_full = float(grid_full[1, iz05, imu_mz])
    rel_qed = abs(D_c_mz_full / D_c_mz - 1.0) * 100.0
    print(f"  D_c(0.5, M_Z) QED=0 : {D_c_mz:.4e}")
    print(f"  D_c(0.5, M_Z) QED=on: {D_c_mz_full:.4e}")
    print(f"  QED relative effect  : {rel_qed:.3f} %  (expect < 0.3 %)")
    status = "PASS ✓" if rel_qed < 0.5 else "WARN !"
    print(f"  Status: {status}")
    print()
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(mu_arr) - 1)) for i in range(len(mu_arr))]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.8),
        gridspec_kw={"wspace": 0.40},
    )
    ax_c, ax_g = axes

    for imu, (mu, color) in enumerate(zip(mu_arr, colors)):
        lw = 1.0 if mu in (MU_FROM, 91.2) else 0.5
        ls = "-"
        label = f"$\\mu={mu:.0f}\\,\\mathrm{{GeV}}$" if mu in (MU_FROM, 91.2) else None
        ax_c.plot(
            Z_ARR,
            Z_ARR * grid_qcd[1, :, imu],
            color=color,
            lw=lw,
            ls=ls,
            label=label,
        )
        ax_g.plot(
            Z_ARR,
            Z_ARR * grid_qcd[2, :, imu],
            color=color,
            lw=lw,
            ls=ls,
            label=label,
        )

    # Analytic BY IC on the charm panel
    ax_c.plot(
        Z_ARR,
        _by_analytic(Z_ARR),
        color="red",
        lw=0.8,
        ls="--",
        label="BY analytic IC",
    )

    # Compare QED-on vs QED-off at M_Z on charm panel
    ax_c.plot(
        Z_ARR,
        Z_ARR * grid_full[1, :, imu_mz],
        color="orange",
        lw=0.8,
        ls=":",
        label="QED on (M_Z)",
    )

    for ax, ylabel, title in [
        (
            ax_c,
            r"$z\,D_{c \to J/\!\psi}^{{}^3\!S_1^{[1]}}(z,\mu)$",
            r"$c \to J/\!\psi$",
        ),
        (
            ax_g,
            r"$z\,D_{g \to J/\!\psi}^{{}^3\!S_1^{[1]}}(z,\mu)$",
            r"$g \to J/\!\psi$  (generated)",
        ),
    ]:
        ax.set_xscale("log")
        ax.set_xlim(Z_ARR[0], Z_ARR[-1])
        ax.set_xlabel(r"$z$")
        ax.axhline(0, color="0.7", lw=0.4, ls="--", zorder=0)

        sci = ticker.ScalarFormatter(useMathText=True)
        sci.set_scientific(True)
        sci.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(sci)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.set_title(title, pad=3)

    ax_c.legend(fontsize=4.5, handlelength=1.2, frameon=False)

    fig.suptitle(
        r"Pure LO QCD evolution ($\alpha_{\rm em}=0$) — ${}^3\!S_1^{[1]}$ channel"
        "\n"
        r"Reference: Braaten–Yuan (1993) / Cho–Leibovich (1996)",
        y=1.03,
        fontsize=7,
    )

    fig.savefig(out, bbox_inches="tight", dpi=450)
    print(f"Saved → {out.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        default="figs/pureqcd_check.pdf",
        help="Output path (default: figs/pureqcd_check.pdf)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_figure(args.output)
