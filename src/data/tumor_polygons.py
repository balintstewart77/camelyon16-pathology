"""
Tumor annotation handling for CAMELYON16 dataset.

Each tumor slide has an XML file with polygon annotations tracing
tumor region boundaries. This module handles parsing those files
and calculating tumor overlap for patch labeling.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

from shapely.geometry import Polygon, box
from shapely.validation import make_valid

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import PatchLabelConfig, DEFAULT_CONFIG


def load_tumor_polygons(xml_path: str) -> List[Polygon]:
    """
    Parse CAMELYON16 XML annotation file and extract tumor polygons.
    
    The XML format contains <Annotation> elements, each with a list of
    <Coordinate> elements defining the polygon vertices.
    
    Invalid polygons (self-intersecting, etc.) are fixed using buffer(0),
    which is a standard Shapely trick for polygon repair.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        List of valid Shapely Polygon objects
        
    Example:
        >>> polygons = load_tumor_polygons('tumor_001.xml')
        >>> print(len(polygons))
        5
        >>> print(polygons[0].area)
        1234567.0
    """
    if not xml_path or not os.path.exists(xml_path):
        return []
    
    try:
        tree = ET.parse(xml_path)
        polygons = []
        
        for annotation in tree.getroot().iter("Annotation"):
            # Extract coordinates from XML
            coords = [
                (float(c.get("X")), float(c.get("Y")))
                for c in annotation.iter("Coordinate")
            ]
            
            # Need at least 3 points for a polygon
            if len(coords) < 3:
                continue
            
            try:
                polygon = Polygon(coords)
                
                if polygon.is_valid:
                    polygons.append(polygon)
                else:
                    # Try to fix invalid polygon using buffer(0)
                    # This handles self-intersections and other issues
                    fixed = make_valid(polygon)
                    
                    # make_valid might return a MultiPolygon
                    if hasattr(fixed, 'geoms'):
                        for geom in fixed.geoms:
                            if isinstance(geom, Polygon) and geom.is_valid:
                                polygons.append(geom)
                    elif isinstance(fixed, Polygon) and fixed.is_valid:
                        polygons.append(fixed)
                        
            except Exception:
                continue
        
        return polygons
        
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []


def calculate_tumor_overlap(
    x: int, 
    y: int, 
    polygons: List[Polygon],
    patch_size: int = 224
) -> float:
    """
    Calculate what fraction of a patch overlaps with tumor regions.
    
    Args:
        x: Patch center X coordinate (in slide coordinates)
        y: Patch center Y coordinate (in slide coordinates)
        polygons: List of tumor Polygon objects
        patch_size: Size of the square patch
        
    Returns:
        Overlap fraction between 0.0 and 1.0
        
    Example:
        >>> overlap = calculate_tumor_overlap(1000, 2000, polygons)
        >>> print(f"{overlap:.2%}")
        '75.23%'
    """
    if not polygons:
        return 0.0
    
    # Create patch as a box polygon
    half = patch_size // 2
    patch_box = box(x - half, y - half, x + half, y + half)
    patch_area = patch_size * patch_size
    
    # Sum intersection areas with all tumor polygons
    total_overlap = 0.0
    for polygon in polygons:
        try:
            if polygon.intersects(patch_box):
                intersection = polygon.intersection(patch_box)
                total_overlap += intersection.area
        except Exception:
            continue
    
    # Cap at 1.0 (polygons might overlap)
    return min(total_overlap / patch_area, 1.0)


def classify_patch(
    x: int,
    y: int,
    polygons: List[Polygon],
    patch_size: int = 224,
    config: PatchLabelConfig = None
) -> int:
    """
    Classify a patch into one of three tumor-related classes.
    
    Classes (for patches from tumor slides):
    - 1: Normal tissue (overlap < boundary_threshold)
    - 2: Boundary tissue (boundary_threshold <= overlap < tumor_threshold)
    - 3: Pure tumor (overlap >= tumor_threshold)
    
    Note: Class 0 (normal from normal slides) is handled separately
    since those slides have no tumor annotations.
    
    Args:
        x, y: Patch center coordinates
        polygons: Tumor polygons from XML
        patch_size: Patch size in pixels
        config: Classification thresholds
        
    Returns:
        Class label (1, 2, or 3)
    """
    if config is None:
        config = DEFAULT_CONFIG.patch_label
    
    overlap = calculate_tumor_overlap(x, y, polygons, patch_size)
    
    if overlap < config.zero_tolerance:
        return 1  # Normal tissue on tumor slide
    elif overlap < config.tumor_threshold:
        return 2  # Boundary tissue
    else:
        return 3  # Pure tumor


def get_patch_label_binary(
    x: int,
    y: int,
    polygons: List[Polygon],
    patch_size: int = 224,
    threshold: float = 0.5
) -> int:
    """
    Simple binary label: is this patch tumor or not?
    
    Simpler version of classify_patch for basic experiments.
    
    Args:
        x, y: Patch center coordinates
        polygons: Tumor polygons
        patch_size: Patch size
        threshold: Overlap threshold for tumor label
        
    Returns:
        0 for normal, 1 for tumor
    """
    overlap = calculate_tumor_overlap(x, y, polygons, patch_size)
    return 1 if overlap >= threshold else 0
