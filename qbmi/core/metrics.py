"""Evaluation metrics for QBMI framework.

Implements accuracy metrics for cell assignment validation.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, Optional
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score, davies_bouldin_score


class CentroidError:
    """Calculates centroid error for cell assignment accuracy.
    
    Formula:
        E = sum(sqrt((x_true - x_calc)^2 + (y_true - y_calc)^2 + (z_true - z_calc)^2))
    """
    
    def __init__(self):
        pass
    
    def compute(
        self,
        true_positions: NDArray,
        calculated_positions: NDArray
    ) -> float:
        """Compute total centroid error.
        
        Args:
            true_positions: True molecule positions (n, 3)
            calculated_positions: Calculated positions (n, 3)
            
        Returns:
            Total centroid error
        """
        if len(true_positions) != len(calculated_positions):
            raise ValueError("Position arrays must have same length")
        
        diff = true_positions - calculated_positions
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        
        return np.sum(distances)
    
    def compute_mean(
        self,
        true_positions: NDArray,
        calculated_positions: NDArray
    ) -> float:
        """Compute mean centroid error.
        
        Args:
            true_positions: True positions
            calculated_positions: Calculated positions
            
        Returns:
            Mean centroid error
        """
        if len(true_positions) != len(calculated_positions):
            raise ValueError("Position arrays must have same length")
        
        diff = true_positions - calculated_positions
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        
        return np.mean(distances)
    
    def compute_per_cell(
        self,
        true_positions: NDArray,
        calculated_positions: NDArray,
        cell_ids: NDArray
    ) -> Dict[int, float]:
        """Compute centroid error per cell.
        
        Args:
            true_positions: True positions
            calculated_positions: Calculated positions
            cell_ids: Cell identifiers
            
        Returns:
            Dictionary mapping cell_id to error
        """
        unique_cells = np.unique(cell_ids)
        errors = {}
        
        for cell_id in unique_cells:
            mask = cell_ids == cell_id
            
            if np.sum(mask) > 0:
                error = self.compute(
                    true_positions[mask],
                    calculated_positions[mask]
                )
                errors[int(cell_id)] = error
        
        return errors
    
    def compute_hausdorff_distance(
        self,
        true_positions: NDArray,
        calculated_positions: NDArray
    ) -> float:
        """Compute Hausdorff distance between point sets.
        
        Args:
            true_positions: True positions
            calculated_positions: Calculated positions
            
        Returns:
            Hausdorff distance
        """
        tree1 = cKDTree(true_positions)
        tree2 = cKDTree(calculated_positions)
        
        max_dist1, _ = tree2.query(true_positions, k=1)
        max_dist2, _ = tree1.query(calculated_positions, k=1)
        
        return max(max_dist1.max(), max_dist2.max())


class GeneDensityThreshold:
    """Evaluates gene density for hidden cell discovery.
    
    Formula:
        Gene Density > Threshold → Hidden Cell Detected
    """
    
    def __init__(self, default_threshold: float = 5.0):
        """Initialize density threshold evaluator.
        
        Args:
            default_threshold: Default density threshold
        """
        self.default_threshold = default_threshold
    
    def compute_density(
        self,
        molecule_coords: NDArray,
        radius: float = 5.0
    ) -> NDArray:
        """Compute local gene density around each molecule.
        
        Args:
            molecule_coords: All molecule positions (n, 3)
            radius: Search radius for density calculation
            
        Returns:
            Array of density values
        """
        tree = cKDTree(molecule_coords)
        
        counts_per_point = np.zeros(len(molecule_coords))
        
        for i in range(len(molecule_coords)):
            d = tree.query(molecule_coords[i:i+1], k=len(molecule_coords))[0]
            counts_per_point[i] = np.sum(d <= radius)
        
        volume = 4/3 * np.pi * radius ** 3
        
        densities = counts_per_point / volume
        
        return densities
    
    def find_density_peaks(
        self,
        molecule_coords: NDArray,
        min_density: float = 5.0,
        min_distance: float = 10.0
    ) -> NDArray:
        """Find peaks in gene density (potential hidden cells).
        
        Args:
            molecule_coords: Molecule positions
            min_density: Minimum density for peak
            min_distance: Minimum distance between peaks
            
        Returns:
            Peak positions
        """
        densities = self.compute_density(molecule_coords)
        
        peaks = []
        peak_indices = np.argsort(densities)[::-1]
        
        tree = cKDTree(molecule_coords)
        
        for idx in peak_indices:
            if densities[idx] < min_density:
                break
            
            is_separate = True
            for peak in peaks:
                dist = np.sqrt(np.sum((molecule_coords[idx] - molecule_coords[peak]) ** 2))
                if dist < min_distance:
                    is_separate = False
                    break
            
            if is_separate:
                peaks.append(idx)
        
        return np.array(peaks)
    
    def evaluate_threshold(
        self,
        molecule_coords: NDArray,
        known_cell_centers: NDArray,
        threshold: Optional[float] = None
    ) -> Dict:
        """Evaluate density threshold against known cells.
        
        Args:
            molecule_coords: All molecule positions
            known_cell_centers: Known cell center positions
            threshold: Optional threshold override
            
        Returns:
            Evaluation results
        """
        if threshold is None:
            threshold = self.default_threshold
        
        densities = self.compute_density(molecule_coords)
        
        tree = cKDTree(known_cell_centers)
        distances, _ = tree.query(molecule_coords, k=1)
        
        detected = densities > threshold
        true_positives = np.sum((detected) & (distances < 20.0))
        false_positives = np.sum((detected) & (distances >= 20.0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        return {
            "threshold": threshold,
            "mean_density": np.mean(densities),
            "max_density": np.max(densities),
            "detected_peaks": np.sum(detected),
            "precision": precision
        }


class AssignmentMetrics:
    """Comprehensive metrics for spatial assignment evaluation."""
    
    def __init__(self):
        pass
    
    def compute_silhouette_score(
        self,
        molecule_coords: NDArray,
        cell_assignments: NDArray
    ) -> float:
        """Compute silhouette score for cluster quality.
        
        Args:
            molecule_coords: Molecule positions
            cell_assignments: Cell assignment labels
            
        Returns:
            Silhouette score (-1 to 1)
        """
        valid_mask = cell_assignments >= 0
        
        if len(np.unique(cell_assignments[valid_mask])) < 2:
            return 0.0
        
        return silhouette_score(
            molecule_coords[valid_mask],
            cell_assignments[valid_mask]
        )
    
    def compute_davies_bouldin_score(
        self,
        molecule_coords: NDArray,
        cell_assignments: NDArray
    ) -> float:
        """Compute Davies-Bouldin index for cluster quality.
        
        Args:
            molecule_coords: Molecule positions
            cell_assignments: Cell assignment labels
            
        Returns:
            Davies-Bouldin score (lower is better)
        """
        valid_mask = cell_assignments >= 0
        
        if len(np.unique(cell_assignments[valid_mask])) < 2:
            return float('inf')
        
        return davies_bouldin_score(
            molecule_coords[valid_mask],
            cell_assignments[valid_mask]
        )
    
    def compute_assignment_accuracy(
        self,
        true_assignments: NDArray,
        predicted_assignments: NDArray
    ) -> Dict[str, float]:
        """Compute assignment accuracy metrics.
        
        Args:
            true_assignments: Ground truth cell assignments
            predicted_assignments: Predicted cell assignments
            
        Returns:
            Dictionary of accuracy metrics
        """
        valid_mask = (true_assignments >= 0) & (predicted_assignments >= 0)
        
        accuracy = np.mean(
            true_assignments[valid_mask] == predicted_assignments[valid_mask]
        )
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(
            true_assignments[valid_mask],
            predicted_assignments[valid_mask],
            average='weighted',
            zero_division=0
        )
        
        recall = recall_score(
            true_assignments[valid_mask],
            predicted_assignments[valid_mask],
            average='weighted',
            zero_division=0
        )
        
        f1 = f1_score(
            true_assignments[valid_mask],
            predicted_assignments[valid_mask],
            average='weighted',
            zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
