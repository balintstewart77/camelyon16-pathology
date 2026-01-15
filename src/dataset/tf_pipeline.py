"""
TensorFlow data pipeline for training.

This module provides high-performance data loading using tf.data.
Key features:
- Parallel chunk reading with interleave
- Balanced sampling between classes
- Deterministic validation for stable metrics
- No slide leakage verification
"""

import os
import gc
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tempfile
import shutil

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import TrainingConfig, DEFAULT_CONFIG


def load_chunk_paths(base_path: str) -> List[Tuple[str, int]]:
    """
    Load all chunk file paths with their labels.
    
    Args:
        base_path: Path to dataset directory
        
    Returns:
        List of (chunk_path, label) tuples
    """
    base = Path(base_path)
    chunks = []
    
    # Try 4-class structure first
    class_dirs = {
        'normal_from_normal': 0,
        'normal_from_tumor': 1,
        'boundary_tumor': 2,
        'pure_tumor': 3
    }
    
    found_4class = False
    for class_name, label in class_dirs.items():
        class_dir = base / class_name
        if class_dir.exists():
            found_4class = True
            for chunk_file in class_dir.glob('*.npz'):
                chunks.append((str(chunk_file), label))
    
    if found_4class:
        return chunks
    
    # Fall back to binary structure
    for class_name, label in [('normal', 0), ('tumor', 1)]:
        class_dir = base / class_name
        if class_dir.exists():
            for chunk_file in class_dir.glob('*.npz'):
                chunks.append((str(chunk_file), label))
    
    return chunks


def verify_no_slide_leakage(
    train_files: List[str],
    val_files: List[str]
) -> None:
    """
    Verify no slide appears in both train and validation sets.
    
    Raises ValueError if leakage detected.
    """
    def collect_slides(files):
        slides = set()
        for f in files:
            try:
                with np.load(f) as data:
                    if 'slides' in data:
                        for s in np.unique(data['slides']):
                            if isinstance(s, bytes):
                                s = s.decode()
                            slides.add(str(s))
            except Exception:
                pass
        return slides
    
    train_slides = collect_slides(train_files)
    val_slides = collect_slides(val_files)
    overlap = train_slides & val_slides
    
    if overlap:
        raise ValueError(
            f"Slide leakage detected! {len(overlap)} slides in both sets. "
            f"Examples: {list(overlap)[:3]}"
        )
    
    print(f"✓ No slide leakage ({len(train_slides)} train, {len(val_slides)} val slides)")


def create_chunk_reader(
    shuffle: bool = True,
    max_patches: int = None,
    seed: int = 42
):
    """
    Create a function that reads a single chunk file.
    
    Returns a callable for use with tf.data.Dataset.interleave()
    """
    rng = np.random.default_rng(seed)
    
    def read_chunk(file_path, label, max_patches_tensor):
        """Python function to read chunk (called via tf.py_function)."""
        path = file_path.numpy().decode('utf-8')
        external_label = label.numpy()
        max_p = max_patches_tensor.numpy()
        
        try:
            with np.load(path, mmap_mode="r") as data:
                X = data['X'].astype(np.float32)
                n = len(X)
                
                # Ensure [0, 1] range
                if X.max() > 1.5:
                    X = X / 255.0
                X = np.clip(X, 0.0, 1.0)
                
                # Subsample if needed
                if max_p > 0 and n > max_p:
                    if shuffle:
                        idx = rng.choice(n, max_p, replace=False)
                    else:
                        idx = np.arange(max_p)
                    X = X[idx]
                    n = max_p
                
                # Use external label (for binary remapping)
                y = np.full(n, external_label, dtype=np.int32)
                
                return X, y
                
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.empty((0, 224, 224, 3), np.float32), np.empty(0, np.int32)
        finally:
            # Help garbage collection release chunk memory
            gc.collect()
    
    def tf_read_chunk(file_path, label, max_patches_per_chunk):
        max_p = tf.constant(
            -1 if max_patches_per_chunk is None else int(max_patches_per_chunk),
            dtype=tf.int32
        )
        
        patches, labels = tf.py_function(
            read_chunk,
            [file_path, label, max_p],
            [tf.float32, tf.int32]
        )
        
        patches.set_shape([None, 224, 224, 3])
        labels.set_shape([None])
        
        ds = tf.data.Dataset.from_tensor_slices((patches, labels))
        
        if shuffle:
            ds = ds.shuffle(buffer_size=1024, seed=seed)
        
        return ds
    
    return tf_read_chunk


def create_dataset(
    chunk_files: List[str],
    labels: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
    max_patches_per_chunk: int = None,
    cycle_length: int = 4
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from chunk files.
    
    Args:
        chunk_files: List of chunk file paths
        labels: List of labels for each chunk
        batch_size: Batch size
        shuffle: Whether to shuffle (True for train, False for val)
        max_patches_per_chunk: Memory control
        cycle_length: Chunks to read in parallel
        
    Returns:
        tf.data.Dataset yielding (batch_x, batch_y)
    """
    # Create file dataset
    file_ds = tf.data.Dataset.from_tensor_slices((chunk_files, labels))
    
    if shuffle:
        file_ds = file_ds.shuffle(len(chunk_files), seed=42, reshuffle_each_iteration=True)
    
    # Create chunk reader
    chunk_reader = create_chunk_reader(shuffle=shuffle, max_patches=max_patches_per_chunk)
    
    # Interleave chunks
    # NOTE: Using fixed num_parallel_calls=2 instead of AUTOTUNE to prevent memory leak.
    # AUTOTUNE spawns many parallel workers, each holding chunk data in memory.
    # With py_function + from_tensor_slices, GC doesn't keep up -> linear RAM growth.
    dataset = file_ds.interleave(
        lambda fp, lbl: chunk_reader(fp, lbl, max_patches_per_chunk),
        cycle_length=cycle_length,
        block_length=4,
        num_parallel_calls=2,  # MEMORY FIX: was tf.data.AUTOTUNE
        deterministic=not shuffle
    )
    
    # Shuffle patches if training
    if shuffle:
        dataset = dataset.shuffle(1000, seed=42)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=shuffle)
    dataset = dataset.prefetch(2)  # MEMORY FIX: was tf.data.AUTOTUNE
    
    # Force determinism for validation
    if not shuffle:
        opts = tf.data.Options()
        opts.deterministic = True
        dataset = dataset.with_options(opts)
    
    return dataset


def create_binary_dataset(
    dataset_path: str,
    class_mapping: Dict[int, List[str]],
    experiment_name: str
) -> str:
    """
    Create a temporary binary dataset by symlinking chunks.
    
    Args:
        dataset_path: Path to 4-class dataset
        class_mapping: Maps binary label to source class names
            e.g., {0: ['normal_from_normal'], 1: ['pure_tumor']}
        experiment_name: Name for temporary directory
        
    Returns:
        Path to temporary binary dataset
    """
    source_path = Path(dataset_path)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{experiment_name.replace(' ', '_')}_"))
    
    (temp_dir / 'normal').mkdir()
    (temp_dir / 'tumor').mkdir()
    
    for binary_label, source_classes in class_mapping.items():
        target_dir = temp_dir / ('normal' if binary_label == 0 else 'tumor')
        
        if isinstance(source_classes, str):
            source_classes = [source_classes]
        
        for source_class in source_classes:
            source_dir = source_path / source_class
            if not source_dir.exists():
                continue
            
            for chunk_file in source_dir.glob('*.npz'):
                # Prefix with source class to avoid collisions
                safe_name = f"{source_class}_{chunk_file.name}"
                link_path = target_dir / safe_name
                
                try:
                    os.symlink(str(chunk_file), str(link_path))
                except OSError:
                    shutil.copy2(str(chunk_file), str(link_path))
    
    return str(temp_dir)


def setup_training_pipeline(
    base_path: str,
    config: TrainingConfig = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Set up complete training and validation pipelines.
    
    Args:
        base_path: Path to dataset (binary structure)
        config: Training configuration
        
    Returns:
        (train_dataset, val_dataset, train_steps, val_steps)
    """
    if config is None:
        config = DEFAULT_CONFIG.training
    
    # Load chunks
    all_chunks = load_chunk_paths(base_path)
    
    if not all_chunks:
        raise ValueError(f"No chunks found in {base_path}")
    
    # Stratified split
    train_chunks, val_chunks = train_test_split(
        all_chunks,
        test_size=config.val_split,
        random_state=42,
        stratify=[lbl for _, lbl in all_chunks]
    )
    
    # Separate by class
    train_files = [f for f, _ in train_chunks]
    train_labels = [l for _, l in train_chunks]
    val_files = [f for f, _ in val_chunks]
    val_labels = [l for _, l in val_chunks]
    
    # Verify no leakage
    verify_no_slide_leakage(train_files, val_files)
    
    print(f"Train: {len(train_files)} chunks, Val: {len(val_files)} chunks")
    
    # Create datasets
    train_dataset = create_dataset(
        train_files, train_labels,
        batch_size=config.batch_size,
        shuffle=True,
        max_patches_per_chunk=config.max_patches_per_chunk,
        cycle_length=config.cycle_length
    )
    train_dataset = train_dataset.repeat()
    
    val_dataset = create_dataset(
        val_files, val_labels,
        batch_size=config.batch_size,
        shuffle=False,  # Deterministic validation
        max_patches_per_chunk= None,  
        cycle_length=min(8, len(val_files))
    )
    
    # Estimate steps
    # Rough estimate based on chunk count and max patches
    patches_per_chunk = config.max_patches_per_chunk or 1000
    train_steps = max(100, len(train_files) * patches_per_chunk // config.batch_size)
    val_steps = None  # Let Keras consume full dataset
    
    print(f"Steps: {train_steps} train, auto val")
    
    return train_dataset, val_dataset, train_steps, val_steps
