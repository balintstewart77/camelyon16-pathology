"""
Patch extraction from Whole Slide Images.

This module handles the core task of extracting 224x224 patches from
gigapixel WSI files. The key challenges are:
1. WSIs don't fit in memory - we work with coordinates, not pixels
2. We want patches from tissue, not background
3. For tumor slides, we want to control patch class distribution
"""

import random
from typing import List, Tuple, Dict, Optional, Iterator

import numpy as np
import openslide
from PIL import Image, ImageEnhance

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import SamplingConfig, DataConfig, DEFAULT_CONFIG

from .tissue_mask import get_tissue_mask, get_scaling_factors
from .tumor_polygons import load_tumor_polygons, classify_patch


def sample_grid_coordinates(
    slide: openslide.OpenSlide,
    mask: np.ndarray,
    patch_size: int = 224,
    stride: int = 224
) -> List[Tuple[int, int]]:
    """
    Generate a grid of patch coordinates over tissue regions.
    
    How it works:
    1. Create a grid at thumbnail resolution (fast!)
    2. Check each grid point against tissue mask
    3. Convert valid points to slide (full resolution) coordinates
    
    Args:
        slide: OpenSlide object
        mask: Tissue mask at thumbnail resolution
        patch_size: Patch size in slide coordinates
        stride: Distance between patch centers (smaller = more overlap)
        
    Returns:
        List of (x, y) coordinates in slide space
        
    Example:
        >>> coords = sample_grid_coordinates(slide, mask, stride=112)
        >>> print(len(coords))
        5432
    """
    scale_x, scale_y = get_scaling_factors(slide, mask)
    slide_w, slide_h = slide.dimensions
    mask_h, mask_w = mask.shape
    
    # Convert stride to thumbnail space
    stride_thumb_x = stride / scale_x
    stride_thumb_y = stride / scale_y
    
    # Half patch for bounds checking
    half_patch = patch_size // 2
    
    coordinates = []
    
    # Iterate over grid in thumbnail space
    for x_thumb in np.arange(0, mask_w, stride_thumb_x):
        for y_thumb in np.arange(0, mask_h, stride_thumb_y):
            x_idx = int(x_thumb)
            y_idx = int(y_thumb)
            
            # Check bounds and tissue presence
            if x_idx >= mask_w or y_idx >= mask_h:
                continue
            if not mask[y_idx, x_idx]:
                continue
            
            # Convert to slide coordinates
            x_slide = int(x_thumb * scale_x)
            y_slide = int(y_thumb * scale_y)
            
            # Check patch fits within slide bounds
            if (x_slide - half_patch >= 0 and 
                x_slide + half_patch < slide_w and
                y_slide - half_patch >= 0 and 
                y_slide + half_patch < slide_h):
                coordinates.append((x_slide, y_slide))
    
    return coordinates


def sample_coordinates_by_class(
    slide_path: str,
    xml_path: str,
    config: SamplingConfig = None,
    patch_size: int = 224
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Sample patch coordinates from a tumor slide, organized by class.
    
    Uses different sampling densities for different regions:
    - Normal regions: sparse sampling (tissue is abundant)
    - Boundary regions: dense sampling (rare and informative)
    - Tumor regions: moderate sampling
    
    Args:
        slide_path: Path to WSI file
        xml_path: Path to XML annotation
        config: Sampling density configuration
        patch_size: Patch size
        
    Returns:
        Dictionary mapping class (1, 2, 3) to coordinate lists
        
    Example:
        >>> coords = sample_coordinates_by_class('tumor_001.tif', 'tumor_001.xml')
        >>> print({k: len(v) for k, v in coords.items()})
        {1: 10234, 2: 456, 3: 2345}
    """
    if config is None:
        config = DEFAULT_CONFIG.sampling
    
    # Load slide and annotations
    slide = openslide.OpenSlide(slide_path)
    polygons = load_tumor_polygons(xml_path)
    
    if not polygons:
        slide.close()
        return {1: [], 2: [], 3: []}
    
    try:
        # Get tissue mask
        mask = get_tissue_mask(slide)
        
        # Get all tissue coordinates with finest stride (for classification)
        # We'll later filter by class and resample if needed
        finest_stride = min(config.normal_stride, config.boundary_stride, config.tumor_stride)
        all_coords = sample_grid_coordinates(slide, mask, patch_size, finest_stride)
        
        # Classify each coordinate
        coords_by_class = {1: [], 2: [], 3: []}
        
        for x, y in all_coords:
            label = classify_patch(x, y, polygons, patch_size)
            coords_by_class[label].append((x, y))
        
        # Subsample based on class-specific stride ratios
        # (denser sampling means keep more coordinates)
        strides = {
            1: config.normal_stride,
            2: config.boundary_stride,
            3: config.tumor_stride
        }
        
        result = {}
        for class_id, coords in coords_by_class.items():
            # Calculate subsampling factor based on stride difference
            subsample = strides[class_id] / finest_stride
            keep_ratio = 1.0 / (subsample ** 2)  # 2D grid
            
            n_keep = max(1, int(len(coords) * keep_ratio))
            
            if len(coords) > n_keep:
                random.shuffle(coords)
                result[class_id] = coords[:n_keep]
            else:
                result[class_id] = coords
        
        return result
        
    finally:
        slide.close()


def extract_patch(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    patch_size: int = 224
) -> Image.Image:
    """
    Extract a single patch from a slide.
    
    Args:
        slide: OpenSlide object
        x, y: Patch center coordinates
        patch_size: Patch size
        
    Returns:
        PIL Image (RGB)
    """
    half = patch_size // 2
    patch = slide.read_region(
        (x - half, y - half),  # Top-left corner
        0,                      # Level 0 (full resolution)
        (patch_size, patch_size)
    )
    return patch.convert("RGB")


def preprocess_patch(
    patch: Image.Image,
    augment: bool = False,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a patch for model input.
    
    Optional augmentation:
    - Random horizontal/vertical flips
    - Random 90-degree rotations
    - Brightness jitter
    
    Args:
        patch: PIL Image
        augment: Whether to apply random augmentation
        normalize: Whether to scale to [0, 1]
        
    Returns:
        NumPy array of shape (224, 224, 3), dtype float32
    """
    img = patch
    
    if augment:
        # Random horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random vertical flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Random 90-degree rotation
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            img = img.rotate(angle)
        
        # Random brightness (0.8 to 1.2)
        factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(img).enhance(factor)
    
    # Convert to array
    arr = np.array(img, dtype=np.float32)
    
    if normalize:
        arr = arr / 255.0
    
    return arr


def extract_patches_from_slide(
    slide_path: str,
    coordinates: List[Tuple[int, int]],
    patch_size: int = 224,
    augment: bool = False
) -> Iterator[Tuple[np.ndarray, int, int]]:
    """
    Extract multiple patches from a slide given coordinates.
    
    Generator that yields patches one at a time to manage memory.
    
    Args:
        slide_path: Path to WSI file
        coordinates: List of (x, y) patch centers
        patch_size: Patch size
        augment: Whether to augment
        
    Yields:
        (patch_array, x, y) tuples
    """
    slide = openslide.OpenSlide(slide_path)
    
    try:
        for x, y in coordinates:
            try:
                patch_pil = extract_patch(slide, x, y, patch_size)
                patch_array = preprocess_patch(patch_pil, augment=augment)
                yield patch_array, x, y
            except Exception:
                continue
    finally:
        slide.close()
