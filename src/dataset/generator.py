"""
4-Class Chunked Dataset Generator for CAMELYON16.

This module generates training datasets from WSI files with proper
slide-aware chunking to prevent data leakage.

Classes:
    0: Normal tissue from normal slides
    1: Normal tissue from tumor slides
    2: Boundary tissue (partial tumor overlap)
    3: Pure tumor tissue

Key Design Decisions:
    - Slide-aware chunking: No slide is split across chunks
    - This enables leak-free train/val splits at chunk level
    - Chunk size ~1000 patches balances memory and I/O
"""

import gc
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openslide

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import Config, DEFAULT_CONFIG

from src.data import (
    list_s3_files,
    download_file_from_s3,
    cleanup_file
)
from src.data.tissue_mask import get_tissue_mask
from src.data.tumor_polygons import load_tumor_polygons, classify_patch
from src.data.patch_extraction import (
    sample_grid_coordinates,
    sample_coordinates_by_class,
    extract_patch,
    preprocess_patch,
    get_stain_normaliser
)


class FourClassGenerator:
    """
    Generates 4-class patch datasets with slide-aware chunking.
    
    Example:
        >>> generator = FourClassGenerator()
        >>> path = generator.generate(
        ...     class_targets={0: 50000, 1: 25000, 2: 25000, 3: 50000},
        ...     save_path="./data/patches"
        ... )
    """
    
    def __init__(self, config: Config = None, stain_normalise: bool = False, reference_image_path: str = None):
        """
        Initialise generator with configuration.

        Args:
            config: Configuration object (uses defaults if None)
            stain_normalise: Whether to apply Macenko stain normalisation
            reference_image_path: Path to reference image for stain normalisation
        """
        self.config = config or DEFAULT_CONFIG
        self.stain_normalise = stain_normalise
        self.reference_image_path = reference_image_path  # Store for metadata

        # Initialise stain normaliser if requested
        if stain_normalise:
            if reference_image_path is None:
                raise ValueError("reference_image_path required when stain_normalise=True")
            from PIL import Image
            reference = np.array(Image.open(reference_image_path))
            get_stain_normaliser(reference)
            print(f"Stain normalisation enabled with reference: {reference_image_path}")

        # Class name mapping
        self.class_names = self.config.class_names
    
    def _save_chunk(
        self,
        patches: List[Tuple[np.ndarray, int, str, int, int]],
        save_dir: Path,
        class_id: int,
        chunk_num: int
    ) -> None:
        """
        Save a chunk of patches to disk.

        Each chunk contains:
        - X: Patch arrays (N, 224, 224, 3)
        - y: Labels (N,)
        - slides: Slide IDs (N,) for leakage verification
        - coords: (x, y) coordinates (N, 2) for visualisation
        """
        if not patches:
            return

        # Unpack patches
        imgs = [p[0] for p in patches]
        labels = [p[1] for p in patches]
        slides = [p[2] for p in patches]
        coords_x = [p[3] for p in patches]
        coords_y = [p[4] for p in patches]

        # Stack into arrays
        X = np.stack(imgs).astype(np.float32)
        y = np.array(labels, dtype=np.int32)
        slide_array = np.array(slides, dtype='U50')
        coords = np.stack([coords_x, coords_y], axis=1).astype(np.int32)

        # Shuffle within chunk
        perm = np.random.permutation(len(X))
        X, y, slide_array, coords = X[perm], y[perm], slide_array[perm], coords[perm]

        # Save
        class_name = self.class_names[class_id]
        filename = f"{class_name}_chunk_{chunk_num:03d}.npz"
        out_path = save_dir / class_name / filename

        np.savez_compressed(out_path, X=X, y=y, slides=slide_array, coords=coords)

        print(f"    Saved {class_name} chunk {chunk_num}: "
              f"{len(X)} patches from {len(set(slides))} slides")
    
    def _process_normal_slide(
        self,
        slide_filename: str,
        n_patches: int
    ) -> List[Tuple[np.ndarray, int, str, int, int]]:
        """
        Extract normal patches from a normal slide.

        Returns list of (patch_array, label=0, slide_id, x, y) tuples.
        """
        # Download slide
        slide_path = download_file_from_s3(
            self.config.data.s3_images,
            slide_filename,
            self.config.data.temp_dir
        )
        if not slide_path:
            return []

        try:
            slide = openslide.OpenSlide(slide_path)
            slide_id = Path(slide_filename).stem

            # Get tissue mask and sample coordinates
            mask = get_tissue_mask(slide)
            coords = sample_grid_coordinates(
                slide, mask,
                patch_size=self.config.data.patch_size,
                stride=self.config.sampling.normal_stride
            )

            # Randomly select coordinates
            random.shuffle(coords)
            coords = coords[:n_patches]

            # Extract patches
            patches = []
            for x, y in coords:
                try:
                    patch_pil = extract_patch(
                        slide, x, y,
                        self.config.data.patch_size
                    )
                    patch_array = preprocess_patch(patch_pil, augment=False, stain_normalise=self.stain_normalise)
                    patches.append((patch_array, 0, slide_id, x, y))
                except Exception:
                    continue

            slide.close()
            print(f"  {slide_filename}: {len(patches)} normal patches")
            return patches

        finally:
            cleanup_file(slide_path)

    def _process_tumor_slide(
        self,
        slide_filename: str,
        patches_per_class: Dict[int, int]
    ) -> Dict[int, List[Tuple[np.ndarray, int, str, int, int]]]:
        """
        Extract patches from a tumor slide, organised by class.

        Returns dict mapping class (1,2,3) to patch lists with coordinates.
        """
        # Download slide and annotations
        slide_path = download_file_from_s3(
            self.config.data.s3_images,
            slide_filename,
            self.config.data.temp_dir
        )
        xml_filename = Path(slide_filename).stem + '.xml'
        xml_path = download_file_from_s3(
            self.config.data.s3_annotations,
            xml_filename,
            self.config.data.temp_dir
        )

        if not slide_path or not xml_path:
            cleanup_file(slide_path)
            cleanup_file(xml_path)
            return {1: [], 2: [], 3: []}

        try:
            # Sample coordinates by class
            coords_by_class = sample_coordinates_by_class(
                slide_path, xml_path,
                config=self.config.sampling,
                patch_size=self.config.data.patch_size
            )

            slide = openslide.OpenSlide(slide_path)
            slide_id = Path(slide_filename).stem

            # Extract patches for each class
            result = {1: [], 2: [], 3: []}

            for class_id in [1, 2, 3]:
                coords = coords_by_class.get(class_id, [])
                target = patches_per_class.get(class_id, 0)

                random.shuffle(coords)
                coords = coords[:target]

                for x, y in coords:
                    try:
                        patch_pil = extract_patch(
                            slide, x, y,
                            self.config.data.patch_size
                        )
                        patch_array = preprocess_patch(patch_pil, augment=False, stain_normalise=self.stain_normalise)
                        result[class_id].append((patch_array, class_id, slide_id, x, y))
                    except Exception:
                        continue

                print(f"    Class {class_id}: {len(result[class_id])}/{target}")

            slide.close()
            return result

        finally:
            cleanup_file(slide_path)
            cleanup_file(xml_path)
    
    def generate(
        self,
        class_targets: Dict[int, int] = None,
        save_path: str = "./data/patches",
        force_regenerate: bool = False
    ) -> Optional[Path]:
        """
        Generate the 4-class dataset.
        
        Args:
            class_targets: Target patches per class
            save_path: Where to save the dataset
            force_regenerate: Regenerate even if exists
            
        Returns:
            Path to generated dataset
        """
        # Set random seed
        np.random.seed(self.config.dataset.seed)
        random.seed(self.config.dataset.seed)
        
        # Use default targets if not provided
        if class_targets is None:
            class_targets = self.config.dataset.class_targets
        
        # Create directories
        base_path = Path(save_path)
        for class_name in self.class_names.values():
            (base_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        if not force_regenerate:
            existing = sum(
                len(list((base_path / name).glob('*.npz')))
                for name in self.class_names.values()
            )
            if existing > 0:
                print(f"Dataset exists with {existing} chunks. Use force_regenerate=True to recreate.")
                return base_path
        
        # Get slide lists
        print("Fetching slide lists from S3...")
        all_slides = list_s3_files(self.config.data.s3_images, '.tif')
        normal_slides = [f for f in all_slides if 'normal' in f.lower()]
        tumor_slides = [f for f in all_slides if 'tumor' in f.lower()]
        
        print(f"Found {len(normal_slides)} normal, {len(tumor_slides)} tumor slides")
        
        # Calculate patches per slide
        normal_per_slide = math.ceil(class_targets[0] / len(normal_slides))
        tumor_per_slide = {
            1: math.ceil(class_targets[1] / len(tumor_slides)),
            2: math.ceil(class_targets[2] / len(tumor_slides)),
            3: math.ceil(class_targets[3] / len(tumor_slides)),
        }
        
        print(f"\nTargets per slide:")
        print(f"  Normal slides: {normal_per_slide} patches")
        print(f"  Tumor slides: {tumor_per_slide}")
        
        # Initialize tracking
        chunk_size = self.config.dataset.chunk_size
        buffers = {cls: [] for cls in range(4)}
        chunk_counts = {cls: 0 for cls in range(4)}
        total_counts = {cls: 0 for cls in range(4)}
        
        # Process normal slides
        print("\n=== Processing Normal Slides ===")
        for slide_filename in normal_slides:
            if total_counts[0] >= class_targets[0]:
                break
            
            patches = self._process_normal_slide(slide_filename, normal_per_slide)
            buffers[0].extend(patches)
            total_counts[0] += len(patches)
            
            # Save chunk if buffer is full
            if len(buffers[0]) >= chunk_size:
                self._save_chunk(buffers[0], base_path, 0, chunk_counts[0])
                chunk_counts[0] += 1
                buffers[0] = []
                gc.collect()
        
        # Save remaining normal patches
        if buffers[0]:
            self._save_chunk(buffers[0], base_path, 0, chunk_counts[0])
            chunk_counts[0] += 1
            buffers[0] = []
        
        # Process tumor slides
        print("\n=== Processing Tumor Slides ===")
        for slide_filename in tumor_slides:
            if all(total_counts[c] >= class_targets[c] for c in [1, 2, 3]):
                break
            
            print(f"\n{slide_filename}")
            patches_by_class = self._process_tumor_slide(slide_filename, tumor_per_slide)
            
            for class_id in [1, 2, 3]:
                patches = patches_by_class[class_id]
                
                # Limit to target
                need = class_targets[class_id] - total_counts[class_id]
                if len(patches) > need:
                    patches = patches[:need]
                
                buffers[class_id].extend(patches)
                total_counts[class_id] += len(patches)
                
                # Save chunk if buffer is full
                if len(buffers[class_id]) >= chunk_size:
                    self._save_chunk(
                        buffers[class_id], base_path,
                        class_id, chunk_counts[class_id]
                    )
                    chunk_counts[class_id] += 1
                    buffers[class_id] = []
                    gc.collect()
        
        # Save remaining patches
        for class_id in [1, 2, 3]:
            if buffers[class_id]:
                self._save_chunk(
                    buffers[class_id], base_path,
                    class_id, chunk_counts[class_id]
                )
                chunk_counts[class_id] += 1
        
        # Save metadata
        metadata = {
            'class_counts': total_counts,
            'class_targets': class_targets,
            'class_names': self.class_names,
            'chunk_counts': chunk_counts,
            'config': {
                'patch_size': self.config.data.patch_size,
                'chunk_size': chunk_size,
                'seed': self.config.dataset.seed,
                'stain_normalised': self.stain_normalise,
                'reference_image': self.reference_image_path if self.stain_normalise else None
            }
        }
        np.save(base_path / 'metadata.npy', metadata)
        
        # Summary
        print("\n" + "=" * 50)
        print("DATASET GENERATION COMPLETE")
        print("=" * 50)
        for class_id, name in self.class_names.items():
            count = total_counts[class_id]
            target = class_targets[class_id]
            pct = count / target * 100 if target > 0 else 0
            print(f"  {name}: {count:,}/{target:,} ({pct:.1f}%)")
        print(f"\nSaved to: {base_path}")

        return base_path

    def generate_test_dataset(
        self,
        save_path: str = "./data/test_patches",
        class_targets: Dict[int, int] = None
    ) -> Optional[Path]:
        """
        Generate 4-class test dataset with balanced class sampling.

        Uses same approach as training: sample by class to ensure
        enough patches per class for reliable evaluation metrics.

        Args:
            save_path: Where to save
            class_targets: Target patches per class
        """
        if class_targets is None:
            class_targets = {
                0: 28000,   # Normal from Normal (~1.1x factor)
                1: 25000,   # Normal from Tumor
                2: 36000,   # Boundary (rare, ~1.45x oversample)
                3: 50000    # Pure Tumor (~2x oversample)
            }

        np.random.seed(self.config.dataset.seed)
        random.seed(self.config.dataset.seed)

        base_path = Path(save_path)
        for class_name in self.class_names.values():
            (base_path / class_name).mkdir(parents=True, exist_ok=True)

        # Get test slides from S3
        print("Fetching test slide list from S3...")
        all_slides = list_s3_files(self.config.data.s3_images, '.tif')
        test_slides = sorted([f for f in all_slides if 'test' in f.lower()])

        # Separate normal vs tumor test slides (tumor slides have annotations)
        normal_test_slides = []
        tumor_test_slides = []

        print("Checking which test slides have annotations...")
        for slide_name in test_slides:
            xml_name = Path(slide_name).stem + '.xml'
            # Check if annotation exists (download returns None if not found)
            xml_path = download_file_from_s3(
                self.config.data.s3_annotations,
                xml_name,
                self.config.data.temp_dir
            )
            if xml_path:
                tumor_test_slides.append(slide_name)
                cleanup_file(xml_path)
            else:
                normal_test_slides.append(slide_name)

        print(f"Found {len(normal_test_slides)} normal test slides, "
              f"{len(tumor_test_slides)} tumor test slides")

        # Calculate patches per slide
        if len(normal_test_slides) > 0:
            normal_per_slide = math.ceil(class_targets[0] / len(normal_test_slides))
        else:
            normal_per_slide = 0

        if len(tumor_test_slides) > 0:
            tumor_per_slide = {
                1: math.ceil(class_targets[1] / len(tumor_test_slides)),
                2: math.ceil(class_targets[2] / len(tumor_test_slides)),
                3: math.ceil(class_targets[3] / len(tumor_test_slides)),
            }
        else:
            tumor_per_slide = {1: 0, 2: 0, 3: 0}

        print(f"\nTargets per slide:")
        print(f"  Normal test slides: {normal_per_slide} patches (class 0)")
        print(f"  Tumor test slides: {tumor_per_slide}")

        # Initialize tracking
        chunk_size = self.config.dataset.chunk_size
        buffers = {cls: [] for cls in range(4)}
        chunk_counts = {cls: 0 for cls in range(4)}
        total_counts = {cls: 0 for cls in range(4)}

        # Process normal test slides (class 0)
        print("\n=== Processing Normal Test Slides ===")
        for slide_filename in normal_test_slides:
            if total_counts[0] >= class_targets[0]:
                break

            patches = self._process_normal_slide(slide_filename, normal_per_slide)
            buffers[0].extend(patches)
            total_counts[0] += len(patches)

            if len(buffers[0]) >= chunk_size:
                self._save_chunk(buffers[0], base_path, 0, chunk_counts[0])
                chunk_counts[0] += 1
                buffers[0] = []
                gc.collect()

        # Save remaining class 0
        if buffers[0]:
            self._save_chunk(buffers[0], base_path, 0, chunk_counts[0])
            chunk_counts[0] += 1
            buffers[0] = []

        # Process tumor test slides (classes 1, 2, 3)
        print("\n=== Processing Tumor Test Slides ===")
        for slide_filename in tumor_test_slides:
            if all(total_counts[c] >= class_targets[c] for c in [1, 2, 3]):
                break

            print(f"\n{slide_filename}")
            patches_by_class = self._process_tumor_slide(slide_filename, tumor_per_slide)

            for class_id in [1, 2, 3]:
                patches = patches_by_class[class_id]

                need = class_targets[class_id] - total_counts[class_id]
                if len(patches) > need:
                    patches = patches[:need]

                buffers[class_id].extend(patches)
                total_counts[class_id] += len(patches)

                if len(buffers[class_id]) >= chunk_size:
                    self._save_chunk(
                        buffers[class_id], base_path,
                        class_id, chunk_counts[class_id]
                    )
                    chunk_counts[class_id] += 1
                    buffers[class_id] = []
                    gc.collect()

        # Save remaining patches
        for class_id in [1, 2, 3]:
            if buffers[class_id]:
                self._save_chunk(
                    buffers[class_id], base_path,
                    class_id, chunk_counts[class_id]
                )
                chunk_counts[class_id] += 1

        # Save metadata
        metadata = {
            'class_counts': total_counts,
            'class_targets': class_targets,
            'class_names': self.class_names,
            'chunk_counts': chunk_counts,
            'num_normal_slides': len(normal_test_slides),
            'num_tumor_slides': len(tumor_test_slides),
            'config': {
                'patch_size': self.config.data.patch_size,
                'chunk_size': chunk_size,
                'seed': self.config.dataset.seed,
                'stain_normalised': self.stain_normalise,
                'reference_image': self.reference_image_path if self.stain_normalise else None
            }
        }
        np.save(base_path / 'metadata.npy', metadata)

        print("\n" + "=" * 50)
        print("TEST DATASET GENERATION COMPLETE")
        print("=" * 50)
        for class_id, name in self.class_names.items():
            count = total_counts[class_id]
            target = class_targets[class_id]
            pct = count / target * 100 if target > 0 else 0
            print(f"  {name}: {count:,}/{target:,} ({pct:.1f}%)")
        print(f"\nSaved to: {base_path}")

        return base_path


def generate_dataset(
    class_targets: Dict[int, int] = None,
    save_path: str = "./data/patches",
    stain_normalise: bool = False,
    reference_image_path: str = None,
    **kwargs
) -> Optional[Path]:
    """
    Convenience function to generate a 4-class dataset.

    Args:
        class_targets: Target patches per class
        save_path: Where to save
        stain_normalise: Whether to apply Macenko stain normalisation
        reference_image_path: Path to reference image for stain normalisation
        **kwargs: Additional arguments for FourClassGenerator

    Returns:
        Path to dataset
    """
    generator = FourClassGenerator(
        stain_normalise=stain_normalise,
        reference_image_path=reference_image_path,
        **kwargs
    )
    return generator.generate(
        class_targets=class_targets,
        save_path=save_path
    )


def generate_test_dataset(
    class_targets: Dict[int, int] = None,
    save_path: str = "./data/test_patches",
    stain_normalise: bool = False,
    reference_image_path: str = None,
    **kwargs
) -> Optional[Path]:
    """
    Convenience function to generate 4-class test dataset.

    Args:
        class_targets: Target patches per class
        save_path: Where to save the test dataset
        stain_normalise: Whether to apply Macenko stain normalisation
        reference_image_path: Path to reference image for stain normalisation
        **kwargs: Additional arguments for FourClassGenerator

    Returns:
        Path to test dataset
    """
    generator = FourClassGenerator(
        stain_normalise=stain_normalise,
        reference_image_path=reference_image_path,
        **kwargs
    )
    return generator.generate_test_dataset(
        class_targets=class_targets,
        save_path=save_path
    )
