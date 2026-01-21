"""
Visualization utilities for WSI patch extraction.

Provides two visualization styles:
1. Scatter plots - dots marking patch centers (fast, good for overview)
2. Grid rectangles - actual patch boundaries as squares (better for understanding)

Both can be used at full-slide thumbnail scale or zoomed into regions.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import openslide


# ============================================================================
# ZOOM REGION HELPERS
# ============================================================================

def find_zoom_region_by_coords(
    coords: List[Tuple[int, int]],
    region_size: int = 8000
) -> Tuple[int, int, int, int]:
    """
    Find a zoom region centered on the mean of coordinates.
    Good for tumor regions where patches cluster around annotations.
    
    Args:
        coords: List of (x, y) patch coordinates
        region_size: Size of square zoom region in pixels
        
    Returns:
        (x_min, y_min, x_max, y_max) bounding box
    """
    if not coords:
        return None
    
    center_x = int(np.mean([x for x, y in coords]))
    center_y = int(np.mean([y for x, y in coords]))
    half_size = region_size // 2
    
    return (center_x - half_size, center_y - half_size,
            center_x + half_size, center_y + half_size)


def find_dense_tissue_region(
    coords: List[Tuple[int, int]],
    region_size: int = 8000
) -> Tuple[int, int, int, int]:
    """
    Find zoom region with highest patch density.
    Better for normal slides where tissue is spread out.
    
    Args:
        coords: List of (x, y) patch coordinates
        region_size: Size of square zoom region
        
    Returns:
        (x_min, y_min, x_max, y_max) bounding box
    """
    if not coords:
        return None
    
    all_x = [x for x, y in coords]
    all_y = [y for x, y in coords]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    half_size = region_size // 2
    best_region = None
    max_patches_in_region = 0
    
    # Sample candidate centers
    x_candidates = [min_x + (max_x - min_x) * f for f in [0.3, 0.5, 0.7]]
    y_candidates = [min_y + (max_y - min_y) * f for f in [0.3, 0.5, 0.7]]
    
    for center_x in x_candidates:
        for center_y in y_candidates:
            region_patches = [
                (x, y) for x, y in coords
                if abs(x - center_x) <= half_size and abs(y - center_y) <= half_size
            ]
            
            if len(region_patches) > max_patches_in_region:
                max_patches_in_region = len(region_patches)
                best_region = (
                    int(center_x - half_size), int(center_y - half_size),
                    int(center_x + half_size), int(center_y + half_size)
                )
    
    # Fallback to center
    if best_region is None:
        center_x = int(np.mean(all_x))
        center_y = int(np.mean(all_y))
        best_region = (center_x - half_size, center_y - half_size,
                      center_x + half_size, center_y + half_size)
    
    return best_region


# ============================================================================
# SCATTER VISUALIZATION (dots at patch centers)
# ============================================================================

def visualize_patches_scatter(
    slide: openslide.OpenSlide,
    coords_by_class: Dict[int, List[Tuple[int, int]]],
    thumbnail_size: Tuple[int, int] = (512, 512),
    class_colors: Dict[int, str] = None,
    class_labels: Dict[int, str] = None,
    title: str = "Patch Distribution",
    max_points_per_class: int = 500,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Visualize patch centers as scatter points on thumbnail.
    
    Fast visualization good for seeing overall distribution.
    
    Args:
        slide: OpenSlide object
        coords_by_class: Dict mapping class ID to list of (x, y) coordinates
        thumbnail_size: Size of thumbnail for display
        class_colors: Dict mapping class ID to color (default provided)
        class_labels: Dict mapping class ID to label (default provided)
        title: Plot title
        max_points_per_class: Limit points per class for speed
        figsize: Figure size
    """
    # Defaults
    if class_colors is None:
        class_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
    if class_labels is None:
        class_labels = {
            0: 'Normal (from normal)',
            1: 'Normal (from tumor)',
            2: 'Boundary',
            3: 'Pure Tumor'
        }
    
    # Get thumbnail and scaling
    thumbnail = slide.get_thumbnail(thumbnail_size)
    slide_w, slide_h = slide.dimensions
    thumb_w, thumb_h = thumbnail.size
    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(thumbnail)
    
    for class_id, coords in sorted(coords_by_class.items()):
        if not coords:
            continue
        
        # Subsample for speed
        if len(coords) > max_points_per_class:
            idx = np.random.choice(len(coords), max_points_per_class, replace=False)
            coords = [coords[i] for i in idx]
        
        # Scale to thumbnail
        x_thumb = [x * scale_x for x, y in coords]
        y_thumb = [y * scale_y for x, y in coords]
        
        color = class_colors.get(class_id, 'gray')
        label = class_labels.get(class_id, f'Class {class_id}')
        n_total = len(coords_by_class[class_id])
        
        ax.scatter(x_thumb, y_thumb, c=color, s=8, alpha=0.7,
                  label=f'{label} ({n_total:,})')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_patches_scatter_zoomed(
    slide: openslide.OpenSlide,
    coords_by_class: Dict[int, List[Tuple[int, int]]],
    zoom_region: Tuple[int, int, int, int],
    class_colors: Dict[int, str] = None,
    class_labels: Dict[int, str] = None,
    title: str = "Zoomed Patch Distribution",
    thumbnail_size: Tuple[int, int] = (2048, 2048),
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Visualize patch centers as scatter points in a zoomed region.
    
    Args:
        slide: OpenSlide object
        coords_by_class: Dict mapping class ID to list of (x, y) coordinates
        zoom_region: (x_min, y_min, x_max, y_max) region to zoom into
        class_colors: Dict mapping class ID to color
        class_labels: Dict mapping class ID to label
        title: Plot title
        thumbnail_size: Max size for zoomed thumbnail
        figsize: Figure size
    """
    if class_colors is None:
        class_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
    if class_labels is None:
        class_labels = {
            0: 'Normal (from normal)',
            1: 'Normal (from tumor)',
            2: 'Boundary',
            3: 'Pure Tumor'
        }
    
    x_min, y_min, x_max, y_max = zoom_region
    region_w = x_max - x_min
    region_h = y_max - y_min
    
    # Choose appropriate level
    level = 0
    while (level < slide.level_count - 1 and
           slide.level_dimensions[level][0] / slide.level_dimensions[0][0] * region_w > thumbnail_size[0]):
        level += 1
    
    downsample = slide.level_downsamples[level]
    thumb_w = min(int(region_w / downsample), thumbnail_size[0])
    thumb_h = min(int(region_h / downsample), thumbnail_size[1])
    
    thumbnail = slide.read_region((x_min, y_min), level, (thumb_w, thumb_h)).convert("RGB")
    
    # Scaling
    scale_x = thumb_w / region_w
    scale_y = thumb_h / region_h
    
    # Filter and scale coordinates
    def filter_and_scale(coords):
        filtered = [(x, y) for x, y in coords if x_min <= x <= x_max and y_min <= y <= y_max]
        return [(int((x - x_min) * scale_x), int((y - y_min) * scale_y)) for x, y in filtered]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(thumbnail)
    
    for class_id, coords in sorted(coords_by_class.items()):
        if not coords:
            continue
        
        display_coords = filter_and_scale(coords)
        if not display_coords:
            continue
        
        x_display, y_display = zip(*display_coords)
        
        color = class_colors.get(class_id, 'gray')
        label = class_labels.get(class_id, f'Class {class_id}')
        
        ax.scatter(x_display, y_display, c=color, s=16, alpha=0.8,
                  label=f'{label} ({len(display_coords)})')
    
    ax.set_title(f"{title}\nRegion: {zoom_region}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# GRID RECTANGLE VISUALIZATION (actual patch boundaries)
# ============================================================================

def visualize_patches_grid(
    slide: openslide.OpenSlide,
    coords_by_class: Dict[int, List[Tuple[int, int]]],
    zoom_region: Tuple[int, int, int, int],
    patch_size: int = 224,
    class_colors: Dict[int, str] = None,
    class_labels: Dict[int, str] = None,
    title: str = "Patch Grid",
    thumbnail_size: Tuple[int, int] = (2048, 2048),
    linewidth: float = 1.0,
    figsize: Tuple[int, int] = (12, 10),
    return_fig: bool = False
) -> None:
    """
    Visualize patches as actual rectangles showing patch boundaries.
    
    This is the "grid" view that shows exactly what patches cover.
    Better for understanding patch extraction but slower than scatter.
    
    Note: Coordinates are assumed to be patch CENTERS. The rectangles
    are drawn from (center - patch_size/2) to show actual coverage.
    
    Args:
        slide: OpenSlide object
        coords_by_class: Dict mapping class ID to list of (x, y) CENTER coordinates
        zoom_region: (x_min, y_min, x_max, y_max) region to zoom into
        patch_size: Size of each patch (default 224)
        class_colors: Dict mapping class ID to color
        class_labels: Dict mapping class ID to label
        title: Plot title
        thumbnail_size: Max size for zoomed thumbnail
        linewidth: Line width for rectangles
        figsize: Figure size
    """
    if class_colors is None:
        class_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
    if class_labels is None:
        class_labels = {
            0: 'Normal (from normal)',
            1: 'Normal (from tumor)',
            2: 'Boundary',
            3: 'Pure Tumor'
        }
    
    x_min, y_min, x_max, y_max = zoom_region
    region_w = x_max - x_min
    region_h = y_max - y_min
    
    # Choose appropriate level
    level = 0
    while (level < slide.level_count - 1 and
           slide.level_dimensions[level][0] / slide.level_dimensions[0][0] * region_w > thumbnail_size[0]):
        level += 1
    
    downsample = slide.level_downsamples[level]
    thumb_w = min(int(region_w / downsample), thumbnail_size[0])
    thumb_h = min(int(region_h / downsample), thumbnail_size[1])
    
    thumbnail = slide.read_region((x_min, y_min), level, (thumb_w, thumb_h)).convert("RGB")
    
    # Scaling
    scale_x = thumb_w / region_w
    scale_y = thumb_h / region_h
    rect_w = patch_size * scale_x
    rect_h = patch_size * scale_y
    half_patch = patch_size // 2
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(thumbnail)
    
    legend_handles = []
    
    for class_id, coords in sorted(coords_by_class.items()):
        if not coords:
            continue
        
        color = class_colors.get(class_id, 'gray')
        label = class_labels.get(class_id, f'Class {class_id}')
        
        # Filter coords that have patches overlapping the zoom region
        # Convert center to top-left for filtering
        filtered = []
        for cx, cy in coords:
            # Top-left of patch
            tl_x = cx - half_patch
            tl_y = cy - half_patch
            # Check if patch overlaps zoom region
            if (tl_x + patch_size > x_min and tl_x < x_max and
                tl_y + patch_size > y_min and tl_y < y_max):
                filtered.append((tl_x, tl_y))
        
        if not filtered:
            continue
        
        # Draw rectangles
        for tl_x, tl_y in filtered:
            # Scale top-left to thumbnail coordinates
            tx = (tl_x - x_min) * scale_x
            ty = (tl_y - y_min) * scale_y
            
            rect = mpatches.Rectangle(
                (tx, ty), rect_w, rect_h,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
        
        # Add to legend
        legend_handles.append(
            mpatches.Patch(edgecolor=color, facecolor='none', 
                          label=f'{label} ({len(filtered)})')
        )
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)
    
    ax.set_title(f"{title}\nRegion: {zoom_region}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    if return_fig:
        return fig
    else:
        plt.show()



def visualize_patches_grid_topleft(
    slide: openslide.OpenSlide,
    coords_by_class: Dict[int, List[Tuple[int, int]]],
    zoom_region: Tuple[int, int, int, int],
    patch_size: int = 224,
    class_colors: Dict[int, str] = None,
    class_labels: Dict[int, str] = None,
    title: str = "Patch Grid",
    thumbnail_size: Tuple[int, int] = (2048, 2048),
    linewidth: float = 1.0,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Visualize patches as rectangles when coordinates are TOP-LEFT (not centers).
    
    Use this version if your coordinates represent the top-left corner of patches
    rather than the center.
    
    Args:
        slide: OpenSlide object
        coords_by_class: Dict mapping class ID to list of (x, y) TOP-LEFT coordinates
        zoom_region: (x_min, y_min, x_max, y_max) region to zoom into
        patch_size: Size of each patch (default 224)
        ... (other args same as visualize_patches_grid)
    """
    if class_colors is None:
        class_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
    if class_labels is None:
        class_labels = {
            0: 'Normal (from normal)',
            1: 'Normal (from tumor)',
            2: 'Boundary',
            3: 'Pure Tumor'
        }
    
    x_min, y_min, x_max, y_max = zoom_region
    region_w = x_max - x_min
    region_h = y_max - y_min
    
    # Choose appropriate level
    level = 0
    while (level < slide.level_count - 1 and
           slide.level_dimensions[level][0] / slide.level_dimensions[0][0] * region_w > thumbnail_size[0]):
        level += 1
    
    downsample = slide.level_downsamples[level]
    thumb_w = min(int(region_w / downsample), thumbnail_size[0])
    thumb_h = min(int(region_h / downsample), thumbnail_size[1])
    
    thumbnail = slide.read_region((x_min, y_min), level, (thumb_w, thumb_h)).convert("RGB")
    
    # Scaling
    scale_x = thumb_w / region_w
    scale_y = thumb_h / region_h
    rect_w = patch_size * scale_x
    rect_h = patch_size * scale_y
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(thumbnail)
    
    legend_handles = []
    
    for class_id, coords in sorted(coords_by_class.items()):
        if not coords:
            continue
        
        color = class_colors.get(class_id, 'gray')
        label = class_labels.get(class_id, f'Class {class_id}')
        
        # Filter: keep patches fully inside zoom window
        filtered = [
            (x, y) for (x, y) in coords
            if (x_min <= x <= x_max - patch_size) and (y_min <= y <= y_max - patch_size)
        ]
        
        if not filtered:
            continue
        
        # Draw rectangles
        for tl_x, tl_y in filtered:
            tx = (tl_x - x_min) * scale_x
            ty = (tl_y - y_min) * scale_y
            
            rect = mpatches.Rectangle(
                (tx, ty), rect_w, rect_h,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
        
        legend_handles.append(
            mpatches.Patch(edgecolor=color, facecolor='none',
                          label=f'{label} ({len(filtered)})')
        )
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)
    
    ax.set_title(f"{title}\nRegion: {zoom_region}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compare_visualization_styles(
    slide: openslide.OpenSlide,
    coords_by_class: Dict[int, List[Tuple[int, int]]],
    zoom_region: Tuple[int, int, int, int],
    patch_size: int = 224,
    title_prefix: str = ""
) -> None:
    """
    Show both scatter and grid visualizations side by side for comparison.
    
    Useful for understanding the difference between the two styles.
    """
    print("Scatter visualization (dots at patch centers):")
    visualize_patches_scatter_zoomed(
        slide, coords_by_class, zoom_region,
        title=f"{title_prefix}Scatter View (patch centers)"
    )
    
    print("\nGrid visualization (actual patch boundaries):")
    visualize_patches_grid(
        slide, coords_by_class, zoom_region, patch_size,
        title=f"{title_prefix}Grid View (patch boundaries)"
    )
