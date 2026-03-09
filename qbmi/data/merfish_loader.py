"""MERFISH data loader for QBMI framework.

Loads and processes MERFISH (Multiplexed Error-Robust Fluorescence in situ
Hybridization) data from the Allen Brain Cell Atlas.
"""

import os
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, List, Tuple, Union
import warnings


class MERFISHDataLoader:
    """Loads MERFISH data from various sources.
    
    Supports:
    - Allen Brain Cell Atlas (ABC Atlas)
    - Local .h5ad files
    - Local .zarr files
    - Custom CSV formats
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize MERFISH data loader.
        
        Args:
            data_path: Path to data directory or file
        """
        self.data_path = data_path
        self.data = None
        self.metadata = {}
    
    def load_from_abc_atlas(
        self,
        dataset: str = "MERFISH Whole Mouse Brain",
        save_dir: Optional[str] = None
    ) -> Dict:
        """Load MERFISH data from Allen Brain Cell Atlas.
        
        Args:
            dataset: Dataset name
            save_dir: Optional directory to save downloaded data
            
        Returns:
            Dictionary containing expression matrix and coordinates
        """
        warnings.warn(
            "Downloading from ABC Atlas requires network access. "
            "This is a template implementation."
        )
        
        return self._create_template_data()
    
    def load_h5ad(
        self,
        file_path: str,
        obsm_key: str = "spatial"
    ) -> Dict:
        """Load MERFISH data from .h5ad file.
        
        Args:
            file_pathh5ad file: Path to .
            obsm_key: Key for spatial coordinates in obsm
            
        Returns:
            Dictionary with data
        """
        try:
            import scanpy as sc
            
            adata = sc.read_h5ad(file_path)
            
            result = {
                "expression_matrix": adata.X,
                "gene_names": adata.var_names.tolist() if hasattr(adata.var_names, 'tolist') else list(adata.var_names),
                "cell_ids": adata.obs_names.tolist() if hasattr(adata.obs_names, 'tolist') else list(adata.obs_names),
            }
            
            if obsm_key in adata.obsm:
                spatial_coords = adata.obsm[obsm_key]
                result["x"] = spatial_coords[:, 0]
                result["y"] = spatial_coords[:, 1]
                if spatial_coords.shape[1] > 2:
                    result["z"] = spatial_coords[:, 2]
            
            self.data = result
            return result
            
        except ImportError:
            warnings.warn("scanpy not installed. Using numpy fallback.")
            return self._load_h5ad_numpy(file_path)
    
    def _load_h5ad_numpy(self, file_path: str) -> Dict:
        """Fallback h5ad loading using h5py."""
        import h5py
        
        result = {}
        
        with h5py.File(file_path, 'r') as f:
            if 'X' in f:
                result['expression_matrix'] = f['X'][:]
            
            if 'obs' in f:
                if 'gene_name' in f['obs']:
                    result['gene_names'] = f['obs/gene_name'][:]
                if 'cell_id' in f['obs']:
                    result['cell_ids'] = f['obs/cell_id'][:]
            
            if 'obsm' in f:
                if 'spatial' in f['obsm']:
                    spatial = f['obsm/spatial'][:]
                    result['x'] = spatial[:, 0]
                    result['y'] = spatial[:, 1]
        
        self.data = result
        return result
    
    def load_zarr(
        self,
        file_path: str
    ) -> Dict:
        """Load MERFISH data from .zarr file.
        
        Args:
            file_path: Path to .zarr directory
            
        Returns:
            Dictionary with data
        """
        try:
            import scanpy as sc
            
            adata = sc.read_zarr(file_path)
            
            result = {
                "expression_matrix": adata.X,
                "gene_names": list(adata.var_names),
                "cell_ids": list(adata.obs_names),
            }
            
            if 'spatial' in adata.obsm:
                spatial = adata.obsm['spatial']
                result["x"] = spatial[:, 0]
                result["y"] = spatial[:, 1]
                if spatial.shape[1] > 2:
                    result["z"] = spatial[:, 2]
            
            self.data = result
            return result
            
        except ImportError:
            warnings.warn("zarr library not installed")
            return {}
    
    def load_csv(
        self,
        expression_file: str,
        coordinates_file: Optional[str] = None,
        gene_column: int = 0,
        sep: str = ","
    ) -> Dict:
        """Load MERFISH data from CSV files.
        
        Args:
            expression_file: Path to expression matrix CSV
            coordinates_file: Optional path to coordinates CSV
            gene_column: Column index for gene names
            sep: CSV separator
            
        Returns:
            Dictionary with data
        """
        import pandas as pd
        
        expr_df = pd.read_csv(expression_file, sep=sep, index_col=0)
        
        result = {
            "expression_matrix": expr_df.values,
            "gene_names": list(expr_df.columns),
            "cell_ids": list(expr_df.index),
        }
        
        if coordinates_file:
            coord_df = pd.read_csv(coordinates_file, sep=sep, index_col=0)
            result["x"] = coord_df["x"].values if "x" in coord_df.columns else coord_df.iloc[:, 0].values
            result["y"] = coord_df["y"].values if "y" in coord_df.columns else coord_df.iloc[:, 1].values
            if "z" in coord_df.columns:
                result["z"] = coord_df["z"].values
        
        self.data = result
        return result
    
    def _create_template_data(
        self,
        n_cells: int = 1000,
        n_genes: int = 300,
        n_molecules: int = 50000
    ) -> Dict:
        """Create template MERFISH-like data for testing.
        
        Args:
            n_cells: Number of cells
            n_genes: Number of genes
            n_molecules: Number of molecules
            
        Returns:
            Dictionary with template data
        """
        np.random.seed(42)
        
        cell_centers = np.random.rand(n_cells, 3) * 1000
        
        cell_radii = np.random.uniform(5, 15, n_cells)
        
        expression_matrix = np.random.poisson(lam=2, size=(n_cells, n_genes))
        
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        cell_ids = [f"Cell_{i}" for i in range(n_cells)]
        
        molecule_coords = []
        molecule_genes = []
        molecule_cells = []
        
        for cell_idx in range(n_cells):
            n_mol = expression_matrix[cell_idx].sum()
            
            for gene_idx in range(n_genes):
                n_gene_mol = expression_matrix[cell_idx, gene_idx]
                
                for _ in range(n_gene_mol):
                    offset = np.random.randn(3) * (cell_radii[cell_idx] / 3)
                    mol_pos = cell_centers[cell_idx] + offset
                    
                    molecule_coords.append(mol_pos)
                    molecule_genes.append(gene_idx)
                    molecule_cells.append(cell_idx)
        
        molecule_coords = np.array(molecule_coords)
        molecule_genes = np.array(molecule_genes)
        molecule_cells = np.array(molecule_cells)
        
        result = {
            "expression_matrix": expression_matrix,
            "gene_names": gene_names,
            "cell_ids": cell_ids,
            "cell_centers": cell_centers,
            "cell_radii": cell_radii,
            "x": cell_centers[:, 0],
            "y": cell_centers[:, 1],
            "z": cell_centers[:, 2],
            "molecule_coords": molecule_coords,
            "molecule_genes": molecule_genes,
            "molecule_cells": molecule_cells,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_molecules": len(molecule_coords)
        }
        
        self.data = result
        self.metadata = {
            "source": "template",
            "dataset": "MERFISH Template"
        }
        
        return result
    
    def get_molecule_positions(self) -> NDArray:
        """Get molecule positions.
        
        Returns:
            Array of molecule coordinates
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        if "molecule_coords" in self.data:
            return self.data["molecule_coords"]
        
        coords = np.column_stack([
            self.data["x"],
            self.data["y"]
        ])
        
        if "z" in self.data:
            coords = np.column_stack([coords, self.data["z"]])
        
        return coords
    
    def get_cell_positions(self) -> NDArray:
        """Get cell center positions.
        
        Returns:
            Array of cell center coordinates
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        if "cell_centers" in self.data:
            return self.data["cell_centers"]
        
        coords = np.column_stack([
            self.data["x"],
            self.data["y"]
        ])
        
        if "z" in self.data:
            coords = np.column_stack([coords, self.data["z"]])
        
        return coords


def load_merfish_example() -> Dict:
    """Load example MERFISH data for testing.
    
    Returns:
        Dictionary with example data
    """
    loader = MERFISHDataLoader()
    return loader._create_template_data()
