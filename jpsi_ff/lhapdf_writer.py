r"""Write evolved J/ψ FFs to LHAPDF grid format.

The set stores  z · D_{a→J/ψ}(z, Q)  for each initial parton *a*, following
the standard LHAPDF convention that grids hold the z-weighted FF.

Parton-ID (PID) mapping
-----------------------
  22 → photon (γ),   4 → charm (c),   21 → gluon (g)

Output structure
----------------
``<output_dir>/<name>/``
  ``<name>.info``        — LHAPDF metadata (YAML-like)
  ``<name>_0000.dat``    — single central member, one grid block

The write logic is a self-contained implementation of the lhagrid1 format,
adapted from ``ekobox.genpdf.export`` (EKO project, MIT licence).

Typical usage
-------------
::

    from jpsi_ff.lhapdf_writer import write_lhapdf_set

    write_lhapdf_set(
        name="JPsi_FF_LO",
        ldmes={"3S18": 1.06e-2, "1S1": 1.16},
        mu_from=3.0,
        mu_values=[3.0, 5.0, 10.0, 91.2],
        alpha_s_ref=0.26,
    )
"""

from __future__ import annotations

import io
import pathlib
import re

import numpy as np
import yaml

from .constants import MC_GEV, MU0_OVER_MC, NF, ALPHA_EM
from .couplings import alpha_s as _alpha_s
from .runner import (
    evolve_ff_grid,
    _default_z_grid,
    DEFAULT_C,
    DEFAULT_T,
    DEFAULT_N_NODES,
)

# ---------------------------------------------------------------------------
# LHAPDF PID convention for the three components
# ---------------------------------------------------------------------------

#: PIDs written to the grid, in component order [γ, c, g].
PIDS: list[int] = [22, 4, 21]
_COMP_OF_PID: dict[int, int] = {22: 0, 4: 1, 21: 2}


# ---------------------------------------------------------------------------
# Low-level LHAPDF format writers  (vendored from ekobox.genpdf.export)
# ---------------------------------------------------------------------------


def _fmt_list(values, fmt: str = "%.6e") -> str:
    return " ".join(fmt % v for v in values)


def _fmt_array(data: np.ndarray) -> str:
    """Serialise a (N, n_pids) data array row-by-row."""
    lines = []
    for row in data:
        lines.append(f"{row[0]:.8e} " + _fmt_list(row[1:], "%.8e"))
    return "\n".join(lines) + "\n"


def _write_dat(path: pathlib.Path, member: int, blocks: list[dict]) -> None:
    """Write one ``.dat`` member file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("PdfType: central\n" if member == 0 else "PdfType: replica\n")
        fh.write("Format: lhagrid1\n---\n")
        for b in blocks:
            fh.write(_fmt_list(b["xgrid"]) + "\n")
            fh.write(_fmt_list(np.sqrt(b["mu2grid"])) + "\n")  # Q, not Q²
            fh.write(_fmt_list(b["pids"], "%d") + "\n")
            fh.write(_fmt_array(b["data"]))
            fh.write("---\n")


def _write_info(path: pathlib.Path, info: dict) -> None:
    """Write the ``.info`` metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    yaml.safe_dump(info, stream, default_flow_style=True, width=100000, line_break="\n")
    # Insert newlines before each top-level key so the file is human-readable
    cnt = re.sub(
        r", ([A-Za-z_]+\d?):(?=[ \[])", r"\n\1:", stream.getvalue().strip()[1:-1]
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(cnt + "\n")


# ---------------------------------------------------------------------------
# Info-file builder
# ---------------------------------------------------------------------------


def build_info(
    set_name: str,
    z_values: list[float],
    mu_values: list[float],
    alpha_s_ref: float,
    mu_from: float,
    mc: float = MC_GEV,
    num_members: int = 1,
    authors: str = "",
    description: str = "",
) -> dict:
    """Build the LHAPDF ``.info`` metadata dictionary.

    Parameters
    ----------
    set_name :
        LHAPDF set name (used in ``SetDesc`` if *description* is empty).
    z_values :
        Sorted z grid.
    mu_values :
        Sorted target μ grid [GeV].
    alpha_s_ref :
        αs(μ_from).
    mu_from :
        Reference scale [GeV] at which *alpha_s_ref* is given.
    mc :
        Charm mass [GeV].
    num_members :
        Number of PDF members in the set.
    authors, description :
        Optional metadata strings.
    """
    mu2_from = mu_from**2
    # Build αs interpolation table covering the full Q range
    q_nodes = sorted({mu_from} | set(mu_values))
    as_vals = [float(_alpha_s(alpha_s_ref, mu2_from, q**2)) for q in q_nodes]

    return {
        "SetDesc": description
        or f"J/psi FF (LO QCD+QED DGLAP, NRQCD ICs) — {set_name}",
        "Authors": authors,
        "Reference": "",
        "Format": "lhagrid1",
        "DataVersion": 1,
        "NumMembers": num_members,
        "Particle": 443,  # J/ψ PDG code
        "Flavors": PIDS,  # [22, 4, 21]
        "ForcePositive": 0,
        "OrderQCD": 1,  # LO
        "FlavorScheme": "fixed",
        "NumFlavors": len(PIDS),
        "XMin": float(min(z_values)),
        "XMax": float(max(z_values)),
        "QMin": round(float(min(mu_values)), 6),
        "QMax": round(float(max(mu_values)), 6),
        "MZ": 91.1876,
        "MCharm": float(mc),
        "AlphaS_MZ": 0.118,
        "AlphaS_OrderQCD": 1,
        "AlphaS_Type": "ipol",
        "AlphaS_Qs": q_nodes,
        "AlphaS_Vals": as_vals,
    }


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------


def build_block(
    z_arr: np.ndarray,
    mu_arr: np.ndarray,
    grid: np.ndarray,
) -> dict:
    """Build the LHAPDF data block from the FF grid.

    Parameters
    ----------
    z_arr : shape (n_z,)
        Momentum-fraction grid.
    mu_arr : shape (n_mu,)
        Scale grid [GeV], sorted ascending.
    grid : shape (3, n_z, n_mu)
        ``grid[comp, iz, imu]`` = D_{a→J/ψ}(z, μ).
        Component order: 0=γ, 1=c, 2=g.

    Returns
    -------
    dict
        ``{xgrid, mu2grid, pids, data}`` ready for :func:`_write_dat`.
        ``data`` has shape ``(n_z * n_mu, n_pids)`` and stores *z · D(z, μ)*.
    """
    n_z, n_mu = len(z_arr), len(mu_arr)
    data = np.empty((n_z * n_mu, len(PIDS)))
    row = 0
    for iz, z in enumerate(z_arr):
        for imu in range(n_mu):
            for jp, pid in enumerate(PIDS):
                comp = _COMP_OF_PID[pid]
                data[row, jp] = z * grid[comp, iz, imu]  # store z·D(z,Q)
            row += 1
    return {
        "xgrid": z_arr.tolist(),
        "mu2grid": (mu_arr**2).tolist(),
        "pids": PIDS,
        "data": data,
    }


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------


def write_lhapdf_set(
    name: str,
    ldmes: dict[str, float],
    mu_from: float,
    mu_values: list[float],
    alpha_s_ref: float,
    z_values: np.ndarray | None = None,
    output_dir: str | pathlib.Path = ".",
    mc: float = MC_GEV,
    mu0_over_mc: float = MU0_OVER_MC,
    alpha_em: float = ALPHA_EM,
    nf: int = NF,
    n_steps: int = 50,
    authors: str = "",
    description: str = "",
    mellin_c: float = DEFAULT_C,
    mellin_T: float = DEFAULT_T,
    mellin_n_nodes: int = DEFAULT_N_NODES,
) -> pathlib.Path:
    """Evolve J/ψ FFs and write an LHAPDF grid set to disk.

    Creates the directory ``<output_dir>/<name>/`` containing:

    * ``<name>.info``      — LHAPDF metadata
    * ``<name>_0000.dat``  — central member grid

    The grid stores  *z · D_{a→J/ψ}(z, Q)*  for *a* ∈ {γ(22), c(4), g(21)}.

    Parameters
    ----------
    name :
        LHAPDF set name (also used as the directory name).
    ldmes :
        ``{channel: value}`` mapping for NRQCD long-distance matrix elements.
        Channels: ``'1S1'``, ``'3S18'``, ``'1S08'``, ``'3PJ8'``.
    mu_from :
        Initial scale μ₀ [GeV].  Should equal ``mu0_over_mc * mc``.
    mu_values :
        Target scales [GeV] at which to tabulate the FFs.
    alpha_s_ref :
        αs(μ_from) used for QCD running.
    z_values :
        z grid.  Defaults to 60 log-spaced points in [0.05, 0.99].
    output_dir :
        Parent directory.  Defaults to the current working directory.
    mc :
        Charm mass [GeV].
    mu0_over_mc :
        Ratio μ₀/mc (used in NRQCD ICs).
    alpha_em :
        QED coupling (fixed throughout the evolution).
    nf :
        Number of active flavours.
    n_steps :
        Integration steps inside the transfer matrix.
    authors, description :
        Free-text fields written to the ``.info`` file.
    mellin_c, mellin_T, mellin_n_nodes :
        Mellin contour parameters.

    Returns
    -------
    pathlib.Path
        Path to the created set directory.
    """
    z_arr = np.asarray(
        z_values if z_values is not None else _default_z_grid(), dtype=float
    )
    mu_arr_sorted = np.sort(np.asarray(mu_values, dtype=float))

    # Compute the (3, n_z, n_mu) grid
    z_out, mu_out, grid = evolve_ff_grid(
        ldmes=ldmes,
        mu_from=mu_from,
        mu_values=mu_arr_sorted.tolist(),
        alpha_s_ref=alpha_s_ref,
        z_values=z_arr,
        mc=mc,
        mu0_over_mc=mu0_over_mc,
        alpha_em=alpha_em,
        nf=nf,
        n_steps=n_steps,
        mellin_c=mellin_c,
        mellin_T=mellin_T,
        mellin_n_nodes=mellin_n_nodes,
    )

    # Build LHAPDF structures
    info = build_info(
        set_name=name,
        z_values=z_out.tolist(),
        mu_values=mu_out.tolist(),
        alpha_s_ref=alpha_s_ref,
        mu_from=mu_from,
        mc=mc,
        authors=authors,
        description=description,
    )
    block = build_block(z_out, mu_out, grid)

    # Write to disk
    target = pathlib.Path(output_dir).resolve() / name
    _write_info(target / f"{name}.info", info)
    _write_dat(target / f"{name}_0000.dat", member=0, blocks=[block])

    return target
