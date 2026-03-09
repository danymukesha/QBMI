"""Core spatial assignment algorithms for QBMI.

This module implements the mathematical framework for assigning
mRNA molecules to cells based on 3D Euclidean distance.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from scipy.spatial import cKDTree
from scipy.stats import norm


class SpatialAssignment:
    """Assigns molecules to cells using 3D Euclidean distance.
    
    Given molecule coordinates (x_m, y_m, z_m) and cell centers
    (x_c, y_c, z_c), assigns each molecule to the nearest cell
    if within the cell radius R.
    
    Formula:
        d = sqrt((x_m - x_c)^2 + (y_m - y_c)^2 + (z_m - z_c)^2)
        If d < R, molecule is assigned to that cell.
    """
    
    def __init__(self, cell_radius: float = 5.0):
        """Initialize spatial assignment.
        
        Args:
            cell_radius: Default radius of cells in micrometers (default: 5.0)
        """
        self.cell_radius = cell_radius
    
    def compute_distance(
        self, 
        molecule_pos: NDArray, 
        cell_center: NDArray
    ) -> NDArray:
        """Compute 3D Euclidean distance between molecule and cell center.
        
        Args:
            molecule_pos: Position of molecule (x, y, z)
            cell_center: Position of cell center (x, y, z)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(np.sum((molecule_pos - cell_center) ** 2))
    
    def assign_molecules(
        self,
        molecule_coords: NDArray,
        cell_centers: NDArray,
        cell_radii: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray]:
        """Assign molecules to nearest cells within radius.
        
        Args:
            molecule_coords: Array of shape (n_molecules, 3) with x, y, z positions
            cell_centers: Array of shape (n_cells, 3) with cell center positions
            cell_radii: Optional array of cell radii (default: uses self.cell_radius)
            
        Returns:
            Tuple of (cell_assignments, distances)
                - cell_assignments: Array of shape (n_molecules,) with assigned cell index
                - distances: Array of shape (n_molecules,) with distance to assigned cell
        """
        if cell_radii is None:
            cell_radii = np.full(len(cell_centers), self.cell_radius)
        
        tree = cKDTree(cell_centers)
        
        distances, nearest_indices = tree.query(molecule_coords)
        
        valid_assignments = distances <= cell_radii[nearest_indices]
        
        assignments = np.full(len(molecule_coords), -1, dtype=int)
        assignments[valid_assignments] = nearest_indices[valid_assignments]
        
        return assignments, distances
    
    def assign_billions_parallel(
        self,
        molecule_coords: NDArray,
        cell_centers: NDArray,
        cell_radii: Optional[NDArray] = None,
        chunk_size: int = 1000000
    ) -> Tuple[NDArray, NDArray]:
        """Process billions of molecules in parallel chunks.
        
        For handling large-scale datasets efficiently.
        
        Args:
            molecule_coords: Array of shape (n_molecules, 3)
            cell_centers: Array of shape (n_cells, 3)
            cell_radii: Optional array of cell radii
            chunk_size: Number of molecules to process per chunk
            
        Returns:
            Tuple of (cell_assignments, distances)
        """
        n_molecules = len(molecule_coords)
        assignments = np.full(n_molecules, -1, dtype=np.int32)
        distances = np.full(n_molecules, np.inf, dtype=np.float32)
        
        for start in range(0, n_molecules, chunk_size):
            end = min(start + chunk_size, n_molecules)
            chunk_assignments, chunk_distances = self.assign_molecules(
                molecule_coords[start:end],
                cell_centers,
                cell_radii
            )
            assignments[start:end] = chunk_assignments
            distances[start:end] = chunk_distances
        
        return assignments, distances


class GaussianKernelAssignment:
    """Assigns molecules using Gaussian kernel probability.
    
    Uses Bayesian inference to assign transcripts to cells when
    cell boundaries are overlapping.
    
    Formula:
        P(m ∈ Cell_i) ∝ exp(-||x_m - μ_i||² / (2σ²))
    
    where x_m is the 3D position and σ is the estimated cell radius.
    """
    
    def __init__(self, sigma: float = 5.0):
        """Initialize Gaussian kernel assignment.
        
        Args:
            sigma: Standard deviation of Gaussian kernel (cell radius estimate)
        """
        self.sigma = sigma
    
    def compute_probability(
        self,
        molecule_pos: NDArray,
        cell_center: NDArray
    ) -> float:
        """Compute probability that molecule belongs to cell.
        
        Args:
            molecule_pos: Position of molecule (x, y, z)
            cell_center: Position of cell center (x, y, z)
            
        Returns:
            Probability value
        """
        distance_sq = np.sum((molecule_pos - cell_center) ** 2)
        return np.exp(-distance_sq / (2 * self.sigma ** 2))
    
    def assign_stochastic(
        self,
        molecule_coords: NDArray,
        cell_centers: NDArray,
        cell_weights: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray]:
        """Stochastically assign molecules based on Gaussian probabilities.
        
        Args:
            molecule_coords: Array of shape (n_molecules, 3)
            cell_centers: Array of shape (n_cells, 3)
            cell_weights: Optional prior weights for each cell
            
        Returns:
            Tuple of (assignments, probabilities)
        """
        n_molecules = len(molecule_coords)
        n_cells = len(cell_centers)
        
        if cell_weights is None:
            cell_weights = np.ones(n_cells)
        
        tree = cKDTree(cell_centers)
        k = min(10, n_cells)
        distances, nearest_indices = tree.query(molecule_coords, k=k)
        
        probabilities = np.zeros((n_molecules, k))
        
        for j in range(k):
            dist_sq = distances[:, j] ** 2
            probabilities[:, j] = np.exp(-dist_sq / (2 * self.sigma ** 2))
        
        probabilities = probabilities * cell_weights[nearest_indices]
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        assignments = nearest_indices[np.arange(n_molecules), 
                                       np.argmax(probabilities, axis=1)]
        
        max_probs = probabilities.max(axis=1)
        
        return assignments, max_probs
    
    def compute_posterior(
        self,
        molecule_coords: NDArray,
        cell_centers: NDArray,
        expression_levels: Optional[NDArray] = None
    ) -> NDArray:
        """Compute posterior probability matrix.
        
        Args:
            molecule_coords: Array of shape (n_molecules, 3)
            cell_centers: Array of shape (n_cells, 3)
            expression_levels: Gene expression levels per cell
            
        Returns:
            Posterior probability matrix of shape (n_molecules, n_cells)
        """
        n_molecules = len(molecule_coords)
        n_cells = len(cell_centers)
        
        posteriors = np.zeros((n_molecules, n_cells))
        
        tree = cKDTree(cell_centers)
        k = min(20, n_cells)
        _, nearest_indices = tree.query(molecule_coords, k=k)
        
        for i in range(n_molecules):
            for j in range(k):
                cell_idx = nearest_indices[i, j]
                distance = np.sqrt(np.sum(
                    (molecule_coords[i] - cell_centers[cell_idx]) ** 2
                ))
                posteriors[i, cell_idx] = np.exp(-distance ** 2 / (2 * self.sigma ** 2))
        
        if expression_levels is not None:
            posteriors = posteriors * expression_levels
        
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
        
        return posteriors
