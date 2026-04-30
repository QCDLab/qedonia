# qedonia

**LO coupled QCD⊗QED DGLAP evolution of quarkonia fragmentation functions**

`qedonia` solves the leading-order DGLAP evolution equations for the
three-component system (γ, c, g) → J/ψ in Mellin space, coupling QCD and
QED mixing at every scale step. NRQCD initial conditions at μ₀ = 2m_c are
provided for the four standard colour-octet/singlet channels. Results can be
written directly to LHAPDF grid sets.

---

## Physics overview

The fragmentation vector **D** = (D_γ, D_c, D_g) satisfies

```text
d D / d ln μ²  =  Γ(N, μ²) · D
```

where the 3×3 anomalous-dimension matrix at LO reads

```text
Γ = ⎡     0          (α/2π) P̃_γc       0         ⎤
    ⎢ (α/2π) P̃_cγ   (αs/2π) P̃_cc   (αs/2π) P̃_cg ⎥
    ⎣     0          (αs/2π) P̃_gc   (αs/2π) P̃_gg  ⎦
```

The evolution operator is computed via the path-ordered iterate-exact method
(matrix exponential at each step midpoint). Mellin inversion uses a
straight-line contour N = c + it with the trapezoidal rule.

Four NRQCD channels are supported as initial conditions at μ₀ = 2m_c:

| Key | Channel | IC type |
|-----|---------|---------|
| `'1S1'`  | ³S₁^[1] | Braaten–Yuan (non-trivial z-shape for c; zero for g) |
| `'3S18'` | ³S₁^[8] | δ(1−z) for both c and g |
| `'1S08'` | ¹S₀^[8] | δ(1−z) for both c and g |
| `'3PJ8'` | ³P_J^[8] | δ(1−z) for both c and g |

---

## Installation

### From source (recommended during development)

```bash
git clone git@github.com:QCDLab/qedonia.git
cd qedonia
pip install -e ".[dev]"
```

The `[dev]` extra installs `pytest` and `matplotlib` for testing and plotting.

### Runtime dependencies

| Package | Version |
|---------|---------|
| Python  | ≥ 3.10  |
| NumPy   | ≥ 1.24  |
| SciPy   | ≥ 1.10  |
| PyYAML  | ≥ 6.0   |

### Verify the installation

```bash
python -m pytest
```

All 66 tests should pass.

---

## Quick start

### Evolve a single channel

```python
from qedonia import evolve_jpsi

# Evolve D_{c → J/ψ} for the ³S₁^[8] channel from μ₀ = 3 GeV to μ = 10 GeV
z, Dc = evolve_jpsi(
    channel="3S18",
    ldme=1.06e-2,       # ⟨O^{J/ψ}(³S₁^[8])⟩  [GeV³]
    mu_from=3.0,        # initial scale μ₀  [GeV]
    mu_to=10.0,         # target scale      [GeV]
    alpha_s_ref=0.26,   # αs(μ₀)
    component="c",      # 'gamma', 'c', or 'g'
)
# z and Dc are 1-D numpy arrays of length 60 (default z-grid)
```

### Sum all channels

```python
from qedonia import evolve_all_channels

ldmes = {
    "1S1":  1.16,     # [GeV³]
    "3S18": 1.06e-2,  # [GeV³]
    "1S08": 8.9e-3,   # [GeV³]
    "3PJ8": 1.6e-2,   # [GeV⁵]
}

z, D_total = evolve_all_channels(
    ldmes=ldmes,
    mu_from=3.0,
    mu_to=10.0,
    alpha_s_ref=0.26,
    component="c",
)
```

### Plot the result

```python
import matplotlib.pyplot as plt
from qedonia import evolve_all_channels

ldmes = {"1S1": 1.16, "3S18": 1.06e-2, "1S08": 8.9e-3, "3PJ8": 1.6e-2}
mu_from = 3.0
alpha_s_ref = 0.26

fig, ax = plt.subplots()
for mu_to, label in [(3.0, "μ = 3 GeV"), (10.0, "μ = 10 GeV"), (91.2, "μ = MZ")]:
    z, D = evolve_all_channels(ldmes, mu_from, mu_to, alpha_s_ref, component="c")
    ax.plot(z, z * D, label=label)

ax.set(xlabel="z", ylabel="z D_{c→J/ψ}(z, μ)", xscale="log")
ax.legend()
plt.tight_layout()
plt.savefig("jpsi_ff_charm.pdf")
```

---

## Generating LHAPDF grids

### Minimal example

```python
from qedonia import write_lhapdf_set

write_lhapdf_set(
    name="JPsi_FF_LO",
    ldmes={"1S1": 1.16, "3S18": 1.06e-2, "1S08": 8.9e-3, "3PJ8": 1.6e-2},
    mu_from=3.0,
    mu_values=[3.0, 5.0, 10.0, 30.0, 91.2],
    alpha_s_ref=0.26,
)
```

This creates the directory `JPsi_FF_LO/` in the current working directory with
two files:

```text
JPsi_FF_LO/
├── JPsi_FF_LO.info        ← LHAPDF metadata
└── JPsi_FF_LO_0000.dat    ← central member grid
```

The grid stores **z · D_{a→J/ψ}(z, Q)** for each initial parton
a ∈ {γ (PID 22), c (PID 4), g (PID 21)}, following the standard LHAPDF
convention. The produced particle is identified as J/ψ via `Particle: 443`
in the info file.

### Full parameter reference

```python
from qedonia import write_lhapdf_set
import numpy as np

write_lhapdf_set(
    # --- identity ---
    name="JPsi_FF_LO",
    description="J/psi FF LO QCD+QED, NRQCD ICs, mc=1.5 GeV",
    authors="A. Author, B. Author",

    # --- NRQCD long-distance matrix elements ---
    ldmes={
        "1S1":  1.16,     # ³S₁^[1]  [GeV³]  color-singlet
        "3S18": 1.06e-2,  # ³S₁^[8]  [GeV³]  color-octet
        "1S08": 8.9e-3,   # ¹S₀^[8]  [GeV³]  color-octet
        "3PJ8": 1.6e-2,   # ³P_J^[8] [GeV⁵]  color-octet
    },

    # --- scales ---
    mu_from=3.0,          # initial scale μ₀ = mu0_over_mc * mc [GeV]
    mu_values=np.geomspace(3.0, 1000.0, 40).tolist(),  # target Q grid

    # --- coupling ---
    alpha_s_ref=0.26,     # αs(μ₀)

    # --- z grid ---
    z_values=np.geomspace(0.02, 0.99, 100).tolist(),

    # --- output location ---
    output_dir="./grids",

    # --- physics parameters ---
    mc=1.5,               # charm mass [GeV]
    mu0_over_mc=2.0,      # μ₀ / mc
    alpha_em=1.0/133.0,   # QED coupling (fixed)
    nf=4,                 # active flavours

    # --- numerics ---
    n_steps=50,           # evolution integration steps
    mellin_c=2.0,         # Mellin contour offset
    mellin_T=60.0,        # Mellin contour upper limit
    mellin_n_nodes=601,   # Mellin quadrature nodes
)
```

### Using the grid with LHAPDF

Once the set is installed in your LHAPDF data path (or the directory is on
`LHAPDF_DATA_PATH`), load it like any other set:

```python
import lhapdf

ff = lhapdf.mkPDF("JPsi_FF_LO", 0)

z, Q = 0.5, 10.0
Dc  = ff.xfxQ(4,  z, Q) / z   # z·D_c  → D_c  (divide out the z factor)
Dg  = ff.xfxQ(21, z, Q) / z
Dga = ff.xfxQ(22, z, Q) / z
```

---

## Lower-level API

### `evolve_ff_grid` — multi-scale grid in one call

`evolve_ff_grid` is the engine behind `write_lhapdf_set`. It returns a
NumPy array so you can post-process the grid before writing it anywhere.

```python
from qedonia import evolve_ff_grid
import numpy as np

z_arr, mu_arr, grid = evolve_ff_grid(
    ldmes={"3S18": 1.06e-2},
    mu_from=3.0,
    mu_values=[3.0, 10.0, 91.2],
    alpha_s_ref=0.26,
    z_values=np.linspace(0.1, 0.9, 50),
)
# grid.shape == (3, 50, 3)  →  (n_comp, n_z, n_mu)
# grid[0]  → D_γ,   grid[1]  → D_c,   grid[2]  → D_g

D_charm_at_MZ = grid[1, :, -1]   # D_c(z, μ = 91.2 GeV)
```

### `evolve_jpsi` — single channel, single scale, single component

Useful for quick scans or cross-checks.

```python
from qedonia import evolve_jpsi

z, D = evolve_jpsi(
    channel="1S1",
    ldme=1.16,
    mu_from=3.0,
    mu_to=91.2,
    alpha_s_ref=0.26,
    component="g",        # gluon FF
)
```

### Building blocks

```python
from qedonia import (
    transfer_matrix,    # 3×3 LO evolution operator E(N; μ₀² → μ²)
    gamma_matrix,       # 3×3 anomalous-dimension matrix Γ(N, μ²)
    initial_vector,     # NRQCD IC vector D̃(N) for one channel
    contour_nodes,      # Mellin contour nodes and weights
    invert_mellin,      # numerical inverse Mellin transform
    alpha_s,            # LO running coupling
    S1,                 # harmonic number S₁(N) via digamma
)
```

---

## Default parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mc` | 1.5 GeV | Charm quark mass |
| `mu0_over_mc` | 2.0 | μ₀ / mc (initial scale ratio) |
| `alpha_em` | 1/133 | QED coupling (fixed) |
| `nf` | 4 | Active flavours (u, d, s, c) |
| `n_steps` | 50 | Evolution integration steps |
| `mellin_c` | 2.0 | Contour offset (right of rightmost pole at N = 1) |
| `mellin_T` | 60.0 | Contour upper limit |
| `mellin_n_nodes` | 601 | Trapezoidal quadrature nodes |
| z grid | 60 log-spaced in [0.05, 0.99] | Default momentum-fraction grid |

---

## Running the tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific module
pytest tests/test_lhapdf_writer.py -v
```
