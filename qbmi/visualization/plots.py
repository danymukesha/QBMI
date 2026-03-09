"""Visualization utilities for QBMI.

Provides plotting functions for spatial transcriptomics data.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Tuple
import warnings


class SpatialPlotter:
    """Creates spatial plots for transcriptomics data."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 100
    ):
        """Initialize spatial plotter.
        
        Args:
            figsize: Figure size in inches
            dpi: Dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_cell_locations(
        self,
        x: NDArray,
        y: NDArray,
        z: Optional[NDArray] = None,
        cell_ids: Optional[NDArray] = None,
        color_by: Optional[str] = None,
        ax=None,
        **kwargs
    ):
        """Plot cell locations in 2D or 3D.
        
        Args:
            x: X coordinates
            y: Y coordinates
            z: Optional Z coordinates
            cell_ids: Cell identifiers
            color_by: Variable to color by
            ax: Optional matplotlib axes
            **kwargs: Additional plotting arguments
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            warnings.warn("matplotlib not installed")
            return
        
        if ax is None:
            if z is not None:
                fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        scatter_kwargs = {
            's': kwargs.get('s', 10),
            'alpha': kwargs.get('alpha', 0.6),
            'c': kwargs.get('c', 'blue')
        }
        
        if z is not None:
            ax.scatter(x, y, z, **scatter_kwargs)
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.set_zlabel('Z (μm)')
        else:
            ax.scatter(x, y, **scatter_kwargs)
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
        
        ax.set_title(kwargs.get('title', 'Cell Locations'))
        
        return ax
    
    def plot_gene_expression(
        self,
        x: NDArray,
        y: NDArray,
        expression: NDArray,
        gene_name: str = "Gene",
        ax=None,
        **kwargs
    ):
        """Plot gene expression as heatmap.
        
        Args:
            x: X coordinates
            y: Y coordinates
            expression: Expression values
            gene_name: Name of gene
            ax: Optional matplotlib axes
            **kwargs: Additional arguments
        """
        try:
            import matplotlib.pyplot as plt
            from scipy.interpolate import griddata
        except ImportError:
            warnings.warn("matplotlib/scipy not installed")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        xi, yi = np.meshgrid(xi, yi)
        
        zi = griddata((x, y), expression, (xi, yi), method='linear')
        
        cmap = kwargs.get('cmap', 'viridis')
        levels = kwargs.get('levels', 20)
        
        contour = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, alpha=0.8)
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(f'{gene_name} Expression')
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'Expression: {gene_name}')
        
        return ax
    
    def plot_molecule_positions(
        self,
        molecule_coords: NDArray,
        color_by: Optional[NDArray] = None,
        ax=None,
        **kwargs
    ):
        """Plot individual molecule positions.
        
        Args:
            molecule_coords: Array of shape (n, 3) or (n, 2)
            color_by: Optional array for coloring
            ax: Optional matplotlib axes
            **kwargs: Additional arguments
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if molecule_coords.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                molecule_coords[:, 0],
                molecule_coords[:, 1],
                molecule_coords[:, 2],
                s=kwargs.get('s', 1),
                c=color_by if color_by is not None else 'red',
                alpha=kwargs.get('alpha', 0.3)
            )
        else:
            ax.scatter(
                molecule_coords[:, 0],
                molecule_coords[:, 1],
                s=kwargs.get('s', 1),
                c=color_by if color_by is not None else 'red',
                alpha=kwargs.get('alpha', 0.3)
            )
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(kwargs.get('title', 'Molecule Positions'))
        
        return ax
    
    def plot_virtual_cells(
        self,
        virtual_cells: Dict,
        ax=None,
        **kwargs
    ):
        """Plot virtual cells with their molecules.
        
        Args:
            virtual_cells: Dictionary of virtual cells
            ax: Optional matplotlib axes
            **kwargs: Additional arguments
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(virtual_cells))))
        
        for i, (cell_id, cell_data) in enumerate(virtual_cells.items()):
            coords = cell_data.get("coordinates", [])
            
            if len(coords) > 0:
                color = colors[i % 20]
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=kwargs.get('s', 5),
                    c=[color],
                    alpha=kwargs.get('alpha', 0.5),
                    label=f"Cell {cell_id}" if i < 10 else None
                )
                
                centroid = cell_data.get("centroid", coords.mean(axis=0))
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    s=100,
                    c=[color],
                    marker='x',
                    linewidths=2
                )
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(kwargs.get('title', 'Virtual Cells'))
        
        if len(virtual_cells) <= 10:
            ax.legend()
        
        return ax
    
    def plot_comparison(
        self,
        true_positions: NDArray,
        predicted_positions: NDArray,
        ax=None
    ) -> Optional[object]:
        """Plot comparison of true vs predicted positions.
        
        Args:
            true_positions: Ground truth positions
            predicted_positions: Predicted positions
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed")
            return
        
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=self.dpi)
            ax1, ax2 = axes
        else:
            ax1 = ax
            ax2 = None
        
        ax1.scatter(
            true_positions[:, 0],
            true_positions[:, 1],
            s=10,
            c='blue',
            alpha=0.5,
            label='True'
        )
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('X (μm)')
        ax1.set_ylabel('Y (μm)')
        
        if ax2:
            ax2.scatter(
                predicted_positions[:, 0],
                predicted_positions[:, 1],
                s=10,
                c='red',
                alpha=0.5,
                label='Predicted'
            )
            ax2.set_title('QBMI Prediction')
            ax2.set_xlabel('X (μm)')
            ax2.set_ylabel('Y (μm)')
        
        return ax1 if ax2 is None else axes


class VolumetricRenderer:
    """Renders 3D volumetric data."""
    
    def __init__(self):
        pass
    
    def create_3d_visualization(
        self,
        molecule_coords: NDArray,
        cell_centers: Optional[NDArray] = None,
        cell_radii: Optional[NDArray] = None,
        output_path: Optional[str] = None
    ) -> None:
        """Create 3D volumetric visualization.
        
        Args:
            molecule_coords: Molecule positions
            cell_centers: Cell center positions
            cell_radii: Cell radii
            output_path: Optional path to save
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            warnings.warn("matplotlib not installed")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(
            molecule_coords[:, 0],
            molecule_coords[:, 1],
            molecule_coords[:, 2],
            s=1,
            c='red',
            alpha=0.3,
            label='mRNA Molecules'
        )
        
        if cell_centers is not None:
            ax.scatter(
                cell_centers[:, 0],
                cell_centers[:, 1],
                cell_centers[:, 2],
                s=50,
                c='blue',
                marker='o',
                label='Cell Centers'
            )
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title('QBMI Volumetric Rendering')
        ax.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_slices(
        self,
        volume: NDArray,
        axis: int = 2,
        n_slices: int = 10,
        figsize: Tuple[int, int] = (15, 3)
    ):
        """Create slices through 3D volume.
        
        Args:
            volume: 3D volume array
            axis: Axis to slice along (0, 1, or 2)
            n_slices: Number of slices
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed")
            return
        
        if axis == 0:
            slices = [volume[i, :, :] for i in np.linspace(0, volume.shape[0]-1, n_slices, dtype=int)]
        elif axis == 1:
            slices = [volume[:, i, :] for i in np.linspace(0, volume.shape[1]-1, n_slices, dtype=int)]
        else:
            slices = [volume[:, :, i] for i in np.linspace(0, volume.shape[2]-1, n_slices, dtype=int)]
        
        fig, axes = plt.subplots(1, n_slices, figsize=figsize)
        
        for i, sl in enumerate(slices):
            axes[i].imshow(sl, cmap='viridis')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return axes
