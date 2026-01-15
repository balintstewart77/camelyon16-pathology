"""
Configuration for CAMELYON16 tumor detection pipeline.

All configurable parameters in one place for easy modification and transparency.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data sources and paths."""
    
    # S3 paths (public, no credentials needed)
    s3_images: str = "s3://camelyon-dataset/CAMELYON16/images/"
    s3_annotations: str = "s3://camelyon-dataset/CAMELYON16/annotations/"
    
    # Local paths
    temp_dir: str = "/tmp"
    
    # Patch extraction
    patch_size: int = 224
    thumbnail_size: tuple = (512, 512)


@dataclass
class TissueMaskConfig:
    """Configuration for tissue detection from thumbnails."""
    
    # Grayscale threshold: pixels darker than this are tissue
    # (background is white ~255, tissue is darker)
    threshold: int = 180
    
    # Minimum blob size in pixels (removes dust/artifacts)
    min_component_size: int = 100
    
    # Maximum hole size to fill (closes small gaps in tissue)
    max_hole_size: int = 100
    
    # Border margin to zero out (removes edge artifacts)
    border_margin: int = 5
    
    # For filtering valid components
    min_blob_area: int = 500
    max_aspect_ratio: float = 5.0  # Removes thin border artifacts


@dataclass  
class PatchLabelConfig:
    """Configuration for labeling patches based on tumor overlap."""
    
    # Overlap thresholds for 4-class classification:
    # Class 0: Normal from normal slides (no tumor annotation)
    # Class 1: Normal from tumor slides (overlap < boundary_threshold)
    # Class 2: Boundary (boundary_threshold <= overlap < tumor_threshold)
    # Class 3: Pure tumor (overlap >= tumor_threshold)
    
    boundary_threshold: float = 0.01   # 1% tumor overlap
    tumor_threshold: float = 0.50      # 50% tumor overlap
    
    # Tolerance for "zero overlap" (floating point safety)
    zero_tolerance: float = 0.00001


@dataclass
class SamplingConfig:
    """Configuration for patch sampling density."""
    
    # Stride between patch centers (smaller = more overlap = more patches)
    # stride == patch_size means no overlap
    normal_stride: int = 224    # No overlap for abundant normal tissue
    boundary_stride: int = 56   # Dense sampling for rare boundary regions
    tumor_stride: int = 112     # Moderate density for tumor regions


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    
    # Target patches per class
    class_targets: Dict[int, int] = field(default_factory=lambda: {
        0: 50000,   # Normal from normal
        1: 25000,   # Normal from tumor
        2: 25000,   # Boundary
        3: 50000,   # Pure tumor
    })
    
    # Chunk size (patches per .npz file)
    # Smaller = less memory, more files
    # Larger = fewer files, but needs more RAM
    chunk_size: int = 1000
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Batch size (limited by GPU memory)
    batch_size: int = 32
    
    # Learning rate
    learning_rate: float = 1e-4
    
    # Maximum epochs (early stopping may end sooner)
    max_epochs: int = 20
    
    # Early stopping patience
    patience: int = 5
    
    # Train/validation split ratio
    val_split: float = 0.3
    
    # Data pipeline
    cycle_length: int = 4      # Chunks to read in parallel
    max_patches_per_chunk: int = 800  # Memory control


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    tissue_mask: TissueMaskConfig = field(default_factory=TissueMaskConfig)
    patch_label: PatchLabelConfig = field(default_factory=PatchLabelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Class name mapping
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "normal_from_normal",
        1: "normal_from_tumor", 
        2: "boundary_tumor",
        3: "pure_tumor"
    })


# Default configuration instance
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the default configuration."""
    return Config()
