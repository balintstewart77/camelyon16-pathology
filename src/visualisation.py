"""
Visualisation utilities for WSI patch extraction.

Provides grid rectangle visualisation showing actual patch boundaries as coloured
squares. Can be used at full-slide thumbnail scale or zoomed into regions.
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
# TISSUE OUTLINE VISUALISATION
# ============================================================================

def visualise_tissue_outline(
    slide: openslide.OpenSlide,
    tissue_mask: np.ndarray,
    tumor_polygons: Optional[List] = None,
    thumbnail_size: Tuple[int, int] = (512, 512),
    outline_colour: str = 'lime',
    outline_width: float = 1.5,
    tumor_colour: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    tumor_alpha: float = 0.5,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    save_as_png: Optional[str] = None
) -> None:
    """
    Visualise tissue regions as an outline contour on the slide thumbnail.

    Optionally overlay tumor regions as semi-transparent filled polygons.
    Useful for blog figures showing tissue detection and tumor annotations.

    Args:
        slide: OpenSlide object
        tissue_mask: Binary tissue mask (from get_tissue_mask or compute_foreground_mask)
        tumor_polygons: Optional list of Shapely Polygon objects for tumor regions
        thumbnail_size: Size of thumbnail to display
        outline_colour: Colour for tissue outline (default: lime green)
        outline_width: Line width for tissue outline
        tumor_colour: RGB tuple for tumor fill (values 0-1)
        tumor_alpha: Transparency of tumor fill (0-1)
        title: Optional plot title
        figsize: Figure size
        save_as_png: If provided, save figure to this filename instead of displaying
    """
    from PIL import Image

    # Get thumbnail
    thumbnail = slide.get_thumbnail(thumbnail_size)
    thumb_w, thumb_h = thumbnail.size

    # Scale factors from level-0 to thumbnail
    slide_w, slide_h = slide.dimensions
    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h

    # Resize mask to thumbnail size
    mask = np.array(tissue_mask).astype(bool)
    if mask.shape[::-1] != (thumb_w, thumb_h):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_resized = mask_img.resize((thumb_w, thumb_h), resample=Image.NEAREST)
        mask_resized = np.array(mask_resized) > 127
    else:
        mask_resized = mask

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(thumbnail)

    # Draw tissue outline as contour
    ax.contour(mask_resized.astype(float), levels=[0.5],
               colors=outline_colour, linewidths=outline_width)

    # Draw tumor regions if provided
    if tumor_polygons:
        for polygon in tumor_polygons:
            try:
                coords = list(polygon.exterior.coords)
                scaled_coords = [(x * scale_x, y * scale_y) for x, y in coords]
                patch = mpatches.Polygon(
                    scaled_coords,
                    closed=True,
                    facecolor=(*tumor_colour, tumor_alpha),
                    edgecolor='none'
                )
                ax.add_patch(patch)
            except Exception:
                continue

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if save_as_png:
        bg = "#f5f5f5"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        fig.savefig(save_as_png, dpi=300, bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        print(f"Saved: {save_as_png}")
    else:
        plt.show()


# ============================================================================
# GRID RECTANGLE VISUALISATION (actual patch boundaries)
# ============================================================================

def visualise_patches_grid(
    slide: openslide.OpenSlide,
    coords_by_class: Dict[int, List[Tuple[int, int]]],
    zoom_region: Tuple[int, int, int, int],
    patch_size: int = 224,
    class_colours: Dict[int, str] = None,
    class_labels: Dict[int, str] = None,
    title: str = "Patch Grid",
    thumbnail_size: Tuple[int, int] = (2048, 2048),
    linewidth: float = 1.0,
    figsize: Tuple[int, int] = (12, 10),
    save_as_png: Optional[str] = None
) -> None:
    """
    Visualise patches as actual rectangles showing patch boundaries.

    This is the "grid" view that shows exactly what patches cover.

    Note: Coordinates are assumed to be patch CENTERS. The rectangles
    are drawn from (center - patch_size/2) to show actual coverage.

    Args:
        slide: OpenSlide object
        coords_by_class: Dict mapping class ID to list of (x, y) CENTER coordinates
        zoom_region: (x_min, y_min, x_max, y_max) region to zoom into
        patch_size: Size of each patch (default 224)
        class_colours: Dict mapping class ID to colour
        class_labels: Dict mapping class ID to label
        title: Plot title
        thumbnail_size: Max size for zoomed thumbnail
        linewidth: Line width for rectangles
        figsize: Figure size
        save_as_png: If provided, save figure to this filename instead of displaying
    """
    if class_colours is None:
        class_colours = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
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

        colour = class_colours.get(class_id, 'gray')
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
                edgecolor=colour,
                facecolor='none'
            )
            ax.add_patch(rect)

        # Add to legend
        legend_handles.append(
            mpatches.Patch(edgecolor=colour, facecolor='none',
                          label=f'{label} ({len(filtered)})')
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

    ax.set_title(f"{title}\nRegion: {zoom_region}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if save_as_png:
        # Set light grey background for clean export
        bg = "#f5f5f5"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        fig.savefig(save_as_png, dpi=300, bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        print(f"Saved: {save_as_png}")
    else:
        plt.show()


def visualise_patches_grid_topleft(
    slide: openslide.OpenSlide,
    coords_by_class: Dict[int, List[Tuple[int, int]]],
    zoom_region: Tuple[int, int, int, int],
    patch_size: int = 224,
    class_colours: Dict[int, str] = None,
    class_labels: Dict[int, str] = None,
    title: str = "Patch Grid",
    thumbnail_size: Tuple[int, int] = (2048, 2048),
    linewidth: float = 1.0,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Visualise patches as rectangles when coordinates are TOP-LEFT (not centers).

    Use this version if your coordinates represent the top-left corner of patches
    rather than the center.

    Args:
        slide: OpenSlide object
        coords_by_class: Dict mapping class ID to list of (x, y) TOP-LEFT coordinates
        zoom_region: (x_min, y_min, x_max, y_max) region to zoom into
        patch_size: Size of each patch (default 224)
        ... (other args same as visualise_patches_grid)
    """
    if class_colours is None:
        class_colours = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
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

        colour = class_colours.get(class_id, 'gray')
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
                edgecolor=colour,
                facecolor='none'
            )
            ax.add_patch(rect)

        legend_handles.append(
            mpatches.Patch(edgecolor=colour, facecolor='none',
                          label=f'{label} ({len(filtered)})')
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

    ax.set_title(f"{title}\nRegion: {zoom_region}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
