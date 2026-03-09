"""Test suite for QBMI core modules."""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')


class TestSpatialAssignment:
    """Tests for SpatialAssignment class."""
    
    def test_compute_distance(self):
        """Test Euclidean distance computation."""
        from qbmi.core.spatial_assignment import SpatialAssignment
        
        assigner = SpatialAssignment(cell_radius=5.0)
        
        molecule = np.array([1.0, 2.0, 3.0])
        cell_center = np.array([4.0, 5.0, 6.0])
        
        distance = assigner.compute_distance(molecule, cell_center)
        
        expected = np.sqrt(3**2 + 3**2 + 3**2)
        assert np.isclose(distance, expected)
    
    def test_assign_molecules(self):
        """Test molecule assignment."""
        from qbmi.core.spatial_assignment import SpatialAssignment
        
        assigner = SpatialAssignment(cell_radius=10.0)
        
        molecules = np.array([
            [0, 0, 0],
            [5, 5, 5],
            [100, 100, 100]
        ])
        
        cells = np.array([
            [0, 0, 0],
            [5, 5, 5]
        ])
        
        assignments, distances = assigner.assign_molecules(molecules, cells)
        
        assert assignments[0] == 0
        assert assignments[1] == 1
        assert assignments[2] == -1
    
    def test_assign_with_radii(self):
        """Test assignment with different cell radii."""
        from qbmi.core.spatial_assignment import SpatialAssignment
        
        assigner = SpatialAssignment()
        
        molecules = np.array([[0, 0, 0], [10, 0, 0]])
        cells = np.array([[0, 0, 0], [10, 0, 0]])
        radii = np.array([5.0, 15.0])
        
        assignments, distances = assigner.assign_molecules(
            molecules, cells, radii
        )
        
        assert assignments[0] == 0
        assert assignments[1] == 1


class TestGaussianKernelAssignment:
    """Tests for GaussianKernelAssignment class."""
    
    def test_compute_probability(self):
        """Test probability computation."""
        from qbmi.core.spatial_assignment import GaussianKernelAssignment
        
        assigner = GaussianKernelAssignment(sigma=5.0)
        
        molecule = np.array([0, 0, 0])
        cell_center = np.array([5, 0, 0])
        
        prob = assigner.compute_probability(molecule, cell_center)
        
        expected = np.exp(-5**2 / (2 * 5**2))
        assert np.isclose(prob, expected)
    
    def test_assign_stochastic(self):
        """Test stochastic assignment."""
        from qbmi.core.spatial_assignment import GaussianKernelAssignment
        
        assigner = GaussianKernelAssignment(sigma=10.0)
        
        molecules = np.random.rand(100, 3) * 100
        cells = np.array([
            [0, 0, 0],
            [50, 50, 50],
            [100, 100, 100]
        ])
        
        assignments, probs = assigner.assign_stochastic(molecules, cells)
        
        assert len(assignments) == 100
        assert len(probs) == 100
        assert np.all((assignments >= 0) & (assignments < 3))
        assert np.all((probs >= 0) & (probs <= 1))


class TestComputationalDeconvolution:
    """Tests for ComputationalDeconvolution class."""
    
    def test_identify_molecules(self):
        """Test molecule identification from image."""
        from qbmi.core.deconvolution import ComputationalDeconvolution
        
        deconv = ComputationalDeconvolution()
        
        image = np.zeros((20, 20, 20))
        image[5, 5, 5] = 1.0
        image[10, 10, 10] = 1.0
        image[15, 15, 15] = 1.0
        
        molecules = deconv.identify_molecules(image, threshold=0.5)
        
        assert len(molecules) >= 3
    
    def test_group_into_virtual_cells(self):
        """Test virtual cell formation."""
        from qbmi.core.deconvolution import ComputationalDeconvolution
        
        deconv = ComputationalDeconvolution(
            min_molecules_per_cell=3,
            epsilon=5.0,
            min_samples=2
        )
        
        coords = np.vstack([
            np.random.randn(20, 3) + [0, 0, 0],
            np.random.randn(20, 3) + [50, 50, 50]
        ])
        
        cells = deconv.group_into_virtual_cells(coords)
        
        assert len(cells) >= 1


class TestCoordinateTransformer:
    """Tests for CoordinateTransformer class."""
    
    def test_pixel_to_physical(self):
        """Test pixel to physical conversion."""
        from qbmi.core.transformations import CoordinateTransformer
        
        transformer = CoordinateTransformer(pixel_size=0.5)
        
        pixel = np.array([[10, 20, 30]])
        
        physical = transformer.pixel_to_physical(pixel)
        
        expected = np.array([[5, 10, 15]])
        assert np.allclose(physical, expected)
    
    def test_apply_translation(self):
        """Test translation."""
        from qbmi.core.transformations import CoordinateTransformer
        
        transformer = CoordinateTransformer()
        
        coords = np.array([[0, 0, 0], [1, 1, 1]])
        translation = np.array([10, 20, 30])
        
        translated = transformer.apply_translation(coords, translation)
        
        expected = np.array([[10, 20, 30], [11, 21, 31]])
        assert np.allclose(translated, expected)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_centroid_error(self):
        """Test centroid error computation."""
        from qbmi.core.metrics import CentroidError
        
        metric = CentroidError()
        
        true_pos = np.array([[0, 0, 0], [10, 0, 0]])
        calc_pos = np.array([[1, 0, 0], [11, 0, 0]])
        
        error = metric.compute(true_pos, calc_pos)
        
        assert np.isclose(error, 2.0)
    
    def test_mean_centroid_error(self):
        """Test mean centroid error."""
        from qbmi.core.metrics import CentroidError
        
        metric = CentroidError()
        
        true_pos = np.array([[0, 0, 0], [10, 0, 0]])
        calc_pos = np.array([[1, 0, 0], [11, 0, 0]])
        
        mean_error = metric.compute_mean(true_pos, calc_pos)
        
        assert np.isclose(mean_error, 1.0)
    
    def test_gene_density(self):
        """Test gene density computation."""
        from qbmi.core.metrics import GeneDensityThreshold
        
        evaluator = GeneDensityThreshold()
        
        coords = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0.2, 0, 0]
        ])
        
        densities = evaluator.compute_density(coords, radius=1.0)
        
        assert len(densities) == 3
        assert np.all(densities > 0)


class TestMERFISHLoader:
    """Tests for MERFISH data loader."""
    
    def test_create_template_data(self):
        """Test template data creation."""
        from qbmi.data.merfish_loader import MERFISHDataLoader
        
        loader = MERFISHDataLoader()
        data = loader._create_template_data(
            n_cells=50,
            n_genes=20,
            n_molecules=500
        )
        
        assert data["n_cells"] == 50
        assert data["n_genes"] == 20
        assert "molecule_coords" in data
        assert "cell_centers" in data
    
    def test_get_molecule_positions(self):
        """Test getting molecule positions."""
        from qbmi.data.merfish_loader import MERFISHDataLoader
        
        loader = MERFISHDataLoader()
        loader._create_template_data(n_cells=10)
        
        positions = loader.get_molecule_positions()
        
        assert positions.shape[1] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
