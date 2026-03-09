# QBMI: Quantum-Barcoded Molecular Imaging

A theoretical framework for spatial transcriptomics that combines the high 
throughput of single-cell sequencing with the spatial context of imaging.

## Overview

Quantum-Barcoded Molecular Imaging (QBMI) solves the "Trade-off Problem" 
between spatial resolution and molecular depth in transcriptomics. 
The framework uses computational algorithms to assign mRNA molecules to cells 
in 3D space without physically separating them.

Here are some key feature of the tool:

- **SpatialAssignment**: to assign molecules to cells using 3D Euclidean distance
- **GaussianKernelAssignment**: to use Bayesian inference for overlapping cell 
boundaries
- **ComputationalDeconvolution**: to group molecules into virtual cells using 
clustering
- **CoordinateTransformer**: to convert pixel data to physical coordinates
- **MERFISH Data Loader**: to support Allen Brain Cell Atlas data

## Installation

```bash
pip install qbmi
```

Or install from source:

```bash
git clone https://github.com/your-repo/qbmi.git
cd qbmi
pip install -e .
```

## Quick Start

```python
import qbmi
from qbmi.data import merfish_loader
from qbmi.core import SpatialAssignment, ComputationalDeconvolution

# Load example MERFISH data
data = merfish_loader.load_merfish_example()

# Initialize spatial assignment
assigner = SpatialAssignment(cell_radius=10.0)

# Assign molecules to cells
assignments, distances = assigner.assign_molecules(
    molecule_coords=data["molecule_coords"],
    cell_centers=data["cell_centers"],
    cell_radii=data["cell_radii"]
)

# Perform computational deconvolution
deconv = ComputationalDeconvolution(
    min_molecules_per_cell=10,
    clustering_algorithm="dbscan",
    epsilon=2.0
)

virtual_cells = deconv.group_into_virtual_cells(
    molecule_coords=data["molecule_coords"],
    gene_ids=data.get("molecule_genes")
)
```

## Mathematical Framework

### Distance-Based Assignment

For a molecule at position $(x_m, y_m, z_m)$ and cell center at $(x_c, y_c, z_c)$:

$$d = \sqrt{(x_m - x_c)^2 + (y_m - y_c)^2 + (z_m - z_c)^2}$$

If $d < R$ (cell radius), the molecule is assigned to that cell.

### Gaussian Kernel Probability

$$P(m \in Cell_i) \propto \exp\left(-\frac{\|x_m - \mu_i\|^2}{2\sigma^2}\right)$$

where $\mu_i$ is the cell center and $\sigma$ is the estimated cell radius.

### Centroid Error

$$E = \sum_{i=1}^{n} \sqrt{(x_{true} - x_{calc})^2 + (y_{true} - y_{calc})^2 + (z_{true} - z_{calc})^2}$$

## Documentation

See the `docs/` directory for detailed documentation.

## Testing

```bash
pytest tests/
```

## License

MIT License
