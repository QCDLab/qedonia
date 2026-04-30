"""Tests for evolve_ff_grid and write_lhapdf_set."""

import tempfile

import numpy as np

from qedonia.runner import evolve_ff_grid
from qedonia.lhapdf_writer import write_lhapdf_set, PIDS, build_block, build_info
from qedonia.constants import MC_GEV, MU0_OVER_MC

_MC = MC_GEV
_MU0 = MU0_OVER_MC * _MC
_AS = 0.26
_LDMES = {"3S18": 1.0}  # unit LDME for easy checking
_Z = np.array([0.3, 0.5, 0.7])
_MUS = [_MU0, 10.0]


# ---------------------------------------------------------------------------
# evolve_ff_grid
# ---------------------------------------------------------------------------


def test_ff_grid_shape():
    z, mu, grid = evolve_ff_grid(
        _LDMES,
        _MU0,
        _MUS,
        _AS,
        z_values=_Z,
        n_steps=10,
        mellin_n_nodes=51,
    )
    assert z.shape == (3,)
    assert mu.shape == (2,)
    assert grid.shape == (3, 3, 2)  # (n_comp, n_z, n_mu)


def test_ff_grid_mu_sorted():
    """Returned mu_arr must be in ascending order regardless of input order."""
    _, mu, _ = evolve_ff_grid(
        _LDMES,
        _MU0,
        [10.0, _MU0],
        _AS,
        z_values=_Z,
        n_steps=10,
        mellin_n_nodes=51,
    )
    assert mu[0] < mu[1]


def test_ff_grid_identity_at_initial_scale():
    """Grid at the initial scale must reproduce evolve_jpsi at the same scale."""
    from qedonia.runner import evolve_jpsi

    z, mu, grid = evolve_ff_grid(
        _LDMES,
        _MU0,
        [_MU0, 10.0],
        _AS,
        z_values=_Z,
        n_steps=20,
        mellin_n_nodes=201,
    )
    _, D_ref = evolve_jpsi(
        "3S18",
        ldme=1.0,
        mu_from=_MU0,
        mu_to=_MU0,
        alpha_s_ref=_AS,
        component="c",
        z_values=_Z,
        n_steps=20,
        mellin_n_nodes=201,
    )
    # Initial-scale slice (imu=0) should agree with single-call result
    np.testing.assert_allclose(grid[1, :, 0], D_ref, rtol=1e-6)


def test_ff_grid_charm_grows_with_scale():
    """The charm FF integrated over z must grow (gluon feeds charm)."""
    z, mu, grid = evolve_ff_grid(
        _LDMES,
        _MU0,
        [_MU0, 30.0],
        _AS,
        z_values=_Z,
        n_steps=30,
        mellin_n_nodes=201,
    )
    charm_low = np.trapezoid(grid[1, :, 0], z)
    charm_high = np.trapezoid(grid[1, :, 1], z)
    assert charm_high > charm_low


# ---------------------------------------------------------------------------
# build_block
# ---------------------------------------------------------------------------


def test_block_structure():
    z, mu, grid = evolve_ff_grid(
        _LDMES,
        _MU0,
        _MUS,
        _AS,
        z_values=_Z,
        n_steps=10,
        mellin_n_nodes=51,
    )
    block = build_block(z, mu, grid)
    assert block["pids"] == PIDS
    assert block["xgrid"] == z.tolist()
    np.testing.assert_allclose(block["mu2grid"], (mu**2).tolist())
    # data rows: n_z * n_mu
    assert block["data"].shape == (len(z) * len(mu), len(PIDS))


def test_block_stores_z_times_D():
    """The block must store z·D, not D."""
    z = np.array([0.5])
    mu = np.array([_MU0])
    grid = np.ones((3, 1, 1))  # D = 1 everywhere
    block = build_block(z, mu, grid)
    # row 0: z[0]*D = 0.5 for all pids
    np.testing.assert_allclose(block["data"][0], [0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# write_lhapdf_set
# ---------------------------------------------------------------------------


def test_write_creates_files():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_lhapdf_set(
            "TestSet",
            _LDMES,
            _MU0,
            _MUS,
            _AS,
            z_values=_Z,
            output_dir=tmp,
            n_steps=10,
            mellin_n_nodes=51,
        )
        assert (path / "TestSet.info").exists()
        assert (path / "TestSet_0000.dat").exists()


def test_info_file_required_keys():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_lhapdf_set(
            "TestSet",
            _LDMES,
            _MU0,
            _MUS,
            _AS,
            z_values=_Z,
            output_dir=tmp,
            n_steps=10,
            mellin_n_nodes=51,
        )
        info_text = (path / "TestSet.info").read_text()
        for key in (
            "Particle",
            "Flavors",
            "Format",
            "NumMembers",
            "OrderQCD",
            "QMin",
            "QMax",
            "XMin",
            "XMax",
        ):
            assert key in info_text, f"Missing key: {key}"


def test_info_particle_jpsi():
    """Particle must be 443 (J/ψ PDG code)."""
    info = build_info("X", [0.1, 0.9], [3.0, 10.0], _AS, _MU0)
    assert info["Particle"] == 443


def test_dat_format_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_lhapdf_set(
            "TestSet",
            _LDMES,
            _MU0,
            _MUS,
            _AS,
            z_values=_Z,
            output_dir=tmp,
            n_steps=10,
            mellin_n_nodes=51,
        )
        lines = (path / "TestSet_0000.dat").read_text().splitlines()
        assert lines[0] == "PdfType: central"
        assert lines[1] == "Format: lhagrid1"
        assert lines[2] == "---"
        # PID line must contain our three PIDs
        pid_line = lines[5]
        assert "22" in pid_line and "4" in pid_line and "21" in pid_line
