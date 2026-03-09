"""Coordinate transformation module for QBMI.

Converts raw pixel data from light-sheet scans into genomic matrices
using transformation matrices.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict
from scipy import ndimage


class CoordinateTransformer:
    """Transforms coordinates between pixel and physical space.
    
    Uses transformation matrix to convert:
        [x' y' z' 1]^T = T * [x y z 1]^T
        
    where T is a 4x4 transformation matrix.
    """
    
    def __init__(
        self,
        pixel_size: float = 0.1,
        transformation_matrix: Optional[NDArray] = None
    ):
        """Initialize coordinate transformer.
        
        Args:
            pixel_size: Size of pixel in micrometers
            transformation_matrix: Optional 4x4 transformation matrix
        """
        self.pixel_size = pixel_size
        
        if transformation_matrix is None:
            self.transformation_matrix = self._create_identity_matrix()
        else:
            self.transformation_matrix = transformation_matrix
    
    def _create_identity_matrix(self) -> NDArray:
        """Create identity transformation matrix."""
        return np.eye(4)
    
    def pixel_to_physical(
        self,
        pixel_coords: NDArray
    ) -> NDArray:
        """Convert pixel coordinates to physical space.
        
        Args:
            pixel_coords: Array of shape (n_points, 3) in pixel space
            
        Returns:
            Array of shape (n_points, 3) in physical space (micrometers)
        """
        n_points = len(pixel_coords)
        
        homogeneous = np.ones((n_points, 4))
        homogeneous[:, :3] = pixel_coords * self.pixel_size
        
        transformed = (self.transformation_matrix @ homogeneous.T).T
        
        return transformed[:, :3]
    
    def physical_to_pixel(
        self,
        physical_coords: NDArray
    ) -> NDArray:
        """Convert physical coordinates to pixel space.
        
        Args:
            physical_coords: Array of shape (n_points, 3) in micrometers
            
        Returns:
            Array of shape (n_points, 3) in pixel space
        """
        n_points = len(physical_coords)
        
        homogeneous = np.ones((n_points, 4))
        homogeneous[:, :3] = physical_coords
        
        inv_matrix = np.linalg.inv(self.transformation_matrix)
        transformed = (inv_matrix @ homogeneous.T).T
        
        return transformed[:, :3] / self.pixel_size
    
    def apply_rotation(
        self,
        coords: NDArray,
        axis: str = "z",
        angle: float = 0.0
    ) -> NDArray:
        """Apply rotation around specified axis.
        
        Args:
            coords: Coordinates to rotate
            axis: Rotation axis ('x', 'y', or 'z')
            angle: Rotation angle in degrees
            
        Returns:
            Rotated coordinates
        """
        angle_rad = np.deg2rad(angle)
        
        if axis == "x":
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == "y":
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        else:
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        
        return (rotation_matrix @ coords.T).T
    
    def apply_translation(
        self,
        coords: NDArray,
        translation: NDArray
    ) -> NDArray:
        """Apply translation to coordinates.
        
        Args:
            coords: Coordinates to translate
            translation: Translation vector (dx, dy, dz)
            
        Returns:
            Translated coordinates
        """
        return coords + translation
    
    def apply_affine_transform(
        self,
        coords: NDArray,
        matrix: NDArray
    ) -> NDArray:
        """Apply arbitrary affine transformation.
        
        Args:
            coords: Input coordinates (n, 3)
            matrix: 4x4 affine transformation matrix
            
        Returns:
            Transformed coordinates
        """
        n_points = len(coords)
        
        homogeneous = np.ones((n_points, 4))
        homogeneous[:, :3] = coords
        
        transformed = (matrix @ homogeneous.T).T
        
        return transformed[:, :3]
    
    def compute_transform_from_points(
        self,
        source_points: NDArray,
        target_points: NDArray
    ) -> NDArray:
        """Compute transformation matrix from corresponding point pairs.
        
        Uses least squares to find optimal transformation.
        
        Args:
            source_points: Source coordinates (n, 3)
            target_points: Target coordinates (n, 3)
            
        Returns:
            4x4 transformation matrix
        """
        from scipy.linalg import lstsq
        
        n = len(source_points)
        
        A = np.zeros((n * 3, 16))
        b = target_points.flatten()
        
        for i in range(n):
            x, y, z = source_points[i]
            row = i * 3
            
            A[row] = [x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            A[row + 1] = [0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            A[row + 2] = [0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0]
        
        result = lstsq(A, b)[0]
        matrix = result.reshape(4, 4)
        
        return matrix
    
    def create_voxel_grid(
        self,
        bounds: Tuple[NDArray, NDArray],
        voxel_size: float = 1.0
    ) -> NDArray:
        """Create 3D voxel grid for volumetric rendering.
        
        Args:
            bounds: Tuple of (min_coords, max_coords)
            voxel_size: Size of each voxel
            
        Returns:
            3D grid of voxel centers
        """
        min_coords, max_coords = bounds
        
        x = np.arange(min_coords[0], max_coords[0], voxel_size)
        y = np.arange(min_coords[1], max_coords[1], voxel_size)
        z = np.arange(min_coords[2], max_coords[2], voxel_size)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        grid = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        return grid


class LightSheetProcessor:
    """Processes light-sheet microscopy data."""
    
    def __init__(
        self,
        wavelength: float = 488.0,
        numerical_aperture: float = 1.1,
        pixel_size: float = 0.1
    ):
        """Initialize light-sheet processor.
        
        Args:
            wavelength: Excitation wavelength in nm
            numerical_aperture: Objective NA
            pixel_size: Pixel size in micrometers
        """
        self.wavelength = wavelength
        self.numerical_aperture = numerical_aperture
        self.pixel_size = pixel_size
    
    def compute_theoretical_resolution(self) -> Dict[str, float]:
        """Compute theoretical diffraction-limited resolution.
        
        Returns:
            Dictionary with xy and z resolution in micrometers
        """
        import numpy as np
        
        lambda_um = self.wavelength / 1000.0
        
        xy_resolution = 0.61 * lambda_um / self.numerical_aperture
        
        z_resolution = 2 * lambda_um / (self.numerical_aperture ** 2)
        
        return {
            "xy_resolution_um": xy_resolution,
            "z_resolution_um": z_resolution
        }
    
    def deconvolve_image(
        self,
        image: NDArray,
        psf: Optional[NDArray] = None,
        iterations: int = 10
    ) -> NDArray:
        """Perform Richardson-Lucy deconvolution.
        
        Args:
            image: Input image stack
            psf: Point spread function (if None, computed theoretically)
            iterations: Number of deconvolution iterations
            
        Returns:
            Deconvolved image
        """
        if psf is None:
            psf = self._compute_psf(image.shape)
        
        deconvolved = image.copy().astype(np.float64)
        psf_flipped = psf[::-1, ::-1, ::-1]
        psf_sum = psf.sum()
        
        for _ in range(iterations):
            convolved = ndimage.convolve(deconvolved, psf, mode='constant')
            
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = image / convolved
                ratio[np.isnan(ratio) | np.isinf(ratio)] = 0
            
            correction = ndimage.convolve(ratio, psf_flipped, mode='constant')
            deconvolved = deconvolved * correction
            deconvolved = np.maximum(deconvolved, 0)
        
        return deconvolved
    
    def _compute_psf(self, shape: Tuple) -> NDArray:
        """Compute theoretical point spread function.
        
        Args:
            shape: Shape of PSF array
            
        Returns:
            3D PSF array
        """
        from scipy.stats import multivariate_normal
        
        center = np.array(shape) / 2
        coords = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        pos = np.stack([c.flatten() for c in coords], axis=1)
        
        resolution = self.compute_theoretical_resolution()
        
        cov = np.diag([
            resolution["z_resolution_um"] ** 2,
            resolution["xy_resolution_um"] ** 2,
            resolution["xy_resolution_um"] ** 2
        ])
        
        rv = multivariate_normal(mean=center, cov=cov)
        psf = rv.pdf(pos).reshape(shape)
        
        return psf / psf.sum()
