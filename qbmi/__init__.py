"""QBMI: Quantum-Barcoded Molecular Imaging Framework

A theoretical framework for spatial transcriptomics that combines
single-cell sequencing with high-resolution spatial imaging.
"""

__version__ = "0.1.0"
__author__ = "QBMI Research Team"

from .core import (
    SpatialAssignment,
    GaussianKernelAssignment, 
    ComputationalDeconvolution,
    NeuralSignalProcessor,
    CoordinateTransformer,
    LightSheetProcessor,
    CentroidError,
    GeneDensityThreshold,
    AssignmentMetrics,
)

__all__ = [
    "SpatialAssignment",
    "GaussianKernelAssignment", 
    "ComputationalDeconvolution",
    "NeuralSignalProcessor",
    "CoordinateTransformer",
    "LightSheetProcessor",
    "CentroidError",
    "GeneDensityThreshold",
    "AssignmentMetrics",
]
