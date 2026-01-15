"""
Tissue mask generation from WSI thumbnails.

The key insight: WSIs have a white background, so tissue appears darker.
We can work at thumbnail resolution (fast!) to identify tissue regions,
then map coordinates back to full resolution for patch extraction.
"""

import numpy as np
import openslide
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import clear_border
from skimage.measure import label as sk_label, regionprops
from typing import Tuple

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])  # Add project root
from config import TissueMaskConfig, DEFAULT_CONFIG


def compute_foreground_mask(
    slide: openslide.OpenSlide,
    config: TissueMaskConfig = None,
    thumb_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Create a binary mask identifying tissue regions in a WSI.
    
    How it works:
    1. Get a small thumbnail of the whole slide
    2. Convert to grayscale (tissue is darker than white background)
    3. Threshold to get binary mask
    4. Clean up with morphological operations
    
    Args:
        slide: OpenSlide object for the WSI
        config: Tissue mask configuration (uses defaults if None)
        thumb_size: Size for thumbnail processing
        
    Returns:
        Binary mask where True = tissue, False = background
        Shape matches the thumbnail size
        
    Example:
        >>> slide = openslide.OpenSlide('tumor_001.tif')
        >>> mask = compute_foreground_mask(slide)
        >>> print(mask.shape, mask.sum())
        (512, 512) 45678
    """
    if config is None:
        config = DEFAULT_CONFIG.tissue_mask
    
    # Step 1: Get thumbnail and convert to grayscale
    # Grayscale values: 0 = black, 255 = white
    thumbnail = slide.get_thumbnail(thumb_size).convert("L")
    thumbnail_array = np.array(thumbnail)
    
    # Step 2: Threshold
    # Tissue is darker than background, so mask where value < threshold
    mask = thumbnail_array < config.threshold
    
    # Step 3: Morphological cleanup
    # Remove tiny specks (dust, artifacts)
    mask = remove_small_objects(mask, min_size=config.min_component_size)
    
    # Remove objects touching the border (often artifacts)
    mask = clear_border(mask)
    
    # Fill small holes within tissue regions
    mask = remove_small_holes(mask, area_threshold=config.max_hole_size)
    
    # Step 4: Zero out border pixels explicitly
    # Sometimes border artifacts survive clear_border
    m = config.border_margin
    if m > 0:
        mask[:m, :] = False
        mask[-m:, :] = False
        mask[:, :m] = False
        mask[:, -m:] = False
    
    return mask.astype(bool)


def filter_valid_components(
    mask: np.ndarray,
    config: TissueMaskConfig = None
) -> np.ndarray:
    """
    Remove invalid tissue blobs based on size and shape.
    
    Why this helps:
    - Border artifacts often appear as thin strips (high aspect ratio)
    - Small blobs are usually noise, not real tissue
    
    Args:
        mask: Binary tissue mask
        config: Configuration with filtering thresholds
        
    Returns:
        Filtered binary mask
    """
    if config is None:
        config = DEFAULT_CONFIG.tissue_mask
    
    # Label connected components (each blob gets unique ID)
    labeled = sk_label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    
    for region in regionprops(labeled):
        area = region.area
        
        # Get bounding box dimensions
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col
        height = max_row - min_row
        
        # Aspect ratio: long side / short side (always >= 1)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        
        # Keep if large enough and not too elongated
        if area >= config.min_blob_area and aspect_ratio <= config.max_aspect_ratio:
            cleaned[labeled == region.label] = True
    
    return cleaned


def get_tissue_mask(
    slide: openslide.OpenSlide,
    config: TissueMaskConfig = None,
    thumb_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Complete tissue mask pipeline: detection + filtering.
    
    This is the main function to use for getting a clean tissue mask.
    
    Args:
        slide: OpenSlide object
        config: Configuration (uses defaults if None)
        thumb_size: Thumbnail size for processing
        
    Returns:
        Clean binary tissue mask
    """
    if config is None:
        config = DEFAULT_CONFIG.tissue_mask
    
    # Get initial mask
    mask = compute_foreground_mask(slide, config, thumb_size)
    
    # Filter out artifacts
    mask = filter_valid_components(mask, config)
    
    return mask


def get_scaling_factors(
    slide: openslide.OpenSlide,
    mask: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate scaling factors from mask coordinates to slide coordinates.
    
    Args:
        slide: OpenSlide object
        mask: Tissue mask (at thumbnail resolution)
        
    Returns:
        (scale_x, scale_y) to multiply mask coords to get slide coords
    """
    slide_w, slide_h = slide.dimensions
    mask_h, mask_w = mask.shape
    
    scale_x = slide_w / mask_w
    scale_y = slide_h / mask_h
    
    return scale_x, scale_y
