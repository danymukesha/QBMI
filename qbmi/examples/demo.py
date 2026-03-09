"""Example script demonstrating QBMI framework.

This script shows how to:
1. Load MERFISH data
2. Perform spatial assignment
3. Run computational deconvolution
4. Evaluate results
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from qbmi.data.merfish_loader import MERFISHDataLoader
from qbmi.core.spatial_assignment import SpatialAssignment, GaussianKernelAssignment
from qbmi.core.deconvolution import ComputationalDeconvolution
from qbmi.core.transformations import CoordinateTransformer, LightSheetProcessor
from qbmi.core.metrics import CentroidError, GeneDensityThreshold, AssignmentMetrics
from qbmi.visualization.plots import SpatialPlotter, VolumetricRenderer


def main():
    print("=" * 60)
    print("QBMI Framework Demonstration")
    print("=" * 60)
    
    print("\n[1] Loading MERFISH data...")
    loader = MERFISHDataLoader()
    data = loader._create_template_data(
        n_cells=500,
        n_genes=100,
        n_molecules=10000
    )
    
    print(f"    Loaded {data['n_cells']} cells")
    print(f"    Loaded {data['n_genes']} genes")
    print(f"    Loaded {data['n_molecules']} molecules")
    
    print("\n[2] Performing spatial assignment...")
    assigner = SpatialAssignment(cell_radius=8.0)
    
    molecule_coords = data["molecule_coords"]
    cell_centers = data["cell_centers"]
    cell_radii = data["cell_radii"]
    
    assignments, distances = assigner.assign_molecules(
        molecule_coords=molecule_coords,
        cell_centers=cell_centers,
        cell_radii=cell_radii
    )
    
    assigned_count = np.sum(assignments >= 0)
    print(f"    Assigned {assigned_count}/{len(molecule_coords)} molecules")
    print(f"    Mean distance: {distances[distances < np.inf].mean():.2f} um")
    
    print("\n[3] Performing Gaussian kernel assignment...")
    gaussian_assigner = GaussianKernelAssignment(sigma=6.0)
    
    ga_assignments, ga_probs = gaussian_assigner.assign_stochastic(
        molecule_coords=molecule_coords,
        cell_centers=cell_centers
    )
    
    print(f"    Stochastic assignments complete")
    print(f"    Mean assignment probability: {ga_probs.mean():.3f}")
    
    print("\n[4] Running computational deconvolution...")
    deconv = ComputationalDeconvolution(
        min_molecules_per_cell=5,
        clustering_algorithm="dbscan",
        epsilon=3.0,
        min_samples=3
    )
    
    virtual_cells = deconv.group_into_virtual_cells(
        molecule_coords=molecule_coords,
        gene_ids=data.get("molecule_genes")
    )
    
    print(f"    Discovered {len(virtual_cells)} virtual cells")
    
    print("\n[5] Discovering hidden cells...")
    hidden_cells = deconv.discover_hidden_cells(
        molecule_coords=molecule_coords,
        known_cell_centers=cell_centers,
        known_cell_radii=cell_radii,
        density_threshold=3.0,
        min_cluster_size=5
    )
    
    print(f"    Found {len(hidden_cells)} potential hidden cells")
    
    print("\n[6] Computing evaluation metrics...")
    centroid_error = CentroidError()
    
    true_positions = molecule_coords[:1000]
    calculated_positions = molecule_coords[:1000] + np.random.randn(1000, 3) * 0.5
    
    mean_error = centroid_error.compute_mean(true_positions, calculated_positions)
    print(f"    Mean centroid error: {mean_error:.2f} um")
    
    density_evaluator = GeneDensityThreshold(default_threshold=2.0)
    densities = density_evaluator.compute_density(molecule_coords, radius=5.0)
    print(f"    Mean gene density: {densities.mean():.4f} molecules/um^3")
    
    assignment_metrics = AssignmentMetrics()
    true_assigns = data["molecule_cells"][:1000]
    
    accuracy = assignment_metrics.compute_assignment_accuracy(
        true_assigns,
        ga_assignments[:1000]
    )
    print(f"    Assignment accuracy: {accuracy['accuracy']:.2%}")
    
    print("\n[7] Coordinate transformation...")
    transformer = CoordinateTransformer(pixel_size=0.1)
    
    pixel_coords = np.random.rand(100, 3) * 1000
    physical_coords = transformer.pixel_to_physical(pixel_coords)
    
    print(f"    Pixel range: {pixel_coords.min():.1f} - {pixel_coords.max():.1f}")
    print(f"    Physical range: {physical_coords.min():.1f} - {physical_coords.max():.1f} um")
    
    print("\n[8] Light-sheet processor...")
    lsp = LightSheetProcessor(
        wavelength=488.0,
        numerical_aperture=1.1,
        pixel_size=0.1
    )
    
    resolution = lsp.compute_theoretical_resolution()
    print(f"    XY resolution: {resolution['xy_resolution_um']:.3f} um")
    print(f"    Z resolution: {resolution['z_resolution_um']:.3f} um")
    
    print("\n" + "=" * 60)
    print("QBMI Demonstration Complete!")
    print("=" * 60)
    
    return {
        "data": data,
        "assignments": assignments,
        "virtual_cells": virtual_cells,
        "hidden_cells": hidden_cells,
        "metrics": {
            "centroid_error": mean_error,
            "accuracy": accuracy,
            "mean_density": densities.mean()
        }
    }


if __name__ == "__main__":
    results = main()
