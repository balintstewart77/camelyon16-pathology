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


def _normalise_patch_channels(patch):
    """Normalise patch to zero mean, unit std per channel."""
    mean = tf.reduce_mean(patch, axis=[0, 1], keepdims=True)
    std = tf.math.reduce_std(patch, axis=[0, 1], keepdims=True) + 1e-7
    return (patch - mean) / std


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
    seed: int = 42
):
    """
    Create a function that reads a single chunk file for interleave.

    Used for training where parallel loading provides better performance
    and natural cross-chunk diversity.

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
                X_mmap = data['X']
                n = len(X_mmap)

                # Determine indices BEFORE loading to preserve mmap benefits
                if max_p > 0 and n > max_p:
                    if shuffle:
                        idx = rng.choice(n, max_p, replace=False)
                    else:
                        idx = np.arange(max_p)
                    idx = np.sort(idx)  # Sequential access is faster for mmap
                else:
                    idx = np.arange(n)

                # Load only selected indices and convert to float32
                X = X_mmap[idx].astype(np.float32)
                n = len(X)

                # Ensure [0, 1] range - check a small sample first
                if X[:min(10, n)].max() > 1.5:
                    X = X / 255.0
                X = np.clip(X, 0.0, 1.0)

                # Use external label (for binary remapping)
                y = np.full(n, external_label, dtype=np.int32)

                return X, y

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.empty((0, 224, 224, 3), np.float32), np.empty(0, np.int32)
        finally:
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


def create_chunk_generator(
    chunk_files: List[str],
    labels: List[int],
    shuffle: bool = True,
    max_patches_per_chunk: int = None,
    seed: int = 42
):
    """
    Create a generator that yields individual patches from chunk files.

    Used for validation where memory safety is critical. Avoids materializing
    entire chunks as tensors, preventing RAM crashes.
    """
    rng = np.random.default_rng(seed)
    file_indices = list(range(len(chunk_files)))

    while True:
        # Shuffle file order each epoch
        if shuffle:
            rng.shuffle(file_indices)

        for file_idx in file_indices:
            path = chunk_files[file_idx]
            external_label = labels[file_idx]

            try:
                with np.load(path, mmap_mode="r") as data:
                    X_mmap = data['X']
                    n = len(X_mmap)

                    # Determine indices BEFORE loading to preserve mmap benefits
                    max_p = max_patches_per_chunk or -1
                    if max_p > 0 and n > max_p:
                        if shuffle:
                            idx = rng.choice(n, max_p, replace=False)
                        else:
                            idx = np.arange(max_p)
                    else:
                        idx = np.arange(n)

                    if shuffle:
                        rng.shuffle(idx)

                    # Check normalisation on first sample
                    needs_normalise = X_mmap[idx[0]].max() > 1.5

                    # Yield patches one at a time - avoids tensor materialisation
                    for i in idx:
                        patch = X_mmap[i].astype(np.float32)
                        if needs_normalise:
                            patch = patch / 255.0
                        patch = np.clip(patch, 0.0, 1.0)
                        yield patch, external_label

            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
            finally:
                gc.collect()

        # For non-training (validation), only go through data once
        if not shuffle:
            break


def _create_single_class_dataset(
    chunk_files: List[str],
    label: int,
    max_patches_per_chunk: int = None,
    cycle_length: int = 4
) -> tf.data.Dataset:
    """
    Create a dataset from chunks of a single class.

    Helper for balanced training - used by create_train_dataset().

    Args:
        chunk_files: List of chunk file paths (all same class)
        label: Class label (0 or 1)
        max_patches_per_chunk: Memory control
        cycle_length: Chunks to read in parallel

    Returns:
        tf.data.Dataset yielding (patch, label) tuples
    """
    # Create labels array matching chunk files
    labels = [label] * len(chunk_files)

    # Create file dataset
    file_ds = tf.data.Dataset.from_tensor_slices((chunk_files, labels))
    file_ds = file_ds.shuffle(len(chunk_files), seed=42, reshuffle_each_iteration=True)

    # Create chunk reader
    chunk_reader = create_chunk_reader(shuffle=True)

    # Interleave chunks for parallel loading
    dataset = file_ds.interleave(
        lambda fp, lbl: chunk_reader(fp, lbl, max_patches_per_chunk),
        cycle_length=cycle_length,
        block_length=4,
        num_parallel_calls=2,
        deterministic=False
    )

    # Shuffle within class and repeat indefinitely
    dataset = dataset.shuffle(500, seed=42)
    dataset = dataset.repeat()

    return dataset


def _shuffle_batch(x, y):
    """Shuffle samples within a batch to mix classes."""
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    return tf.gather(x, idx), tf.gather(y, idx)


def create_train_dataset(
    chunk_files: List[str],
    labels: List[int],
    batch_size: int = 32,
    max_patches_per_chunk: int = None,
    cycle_length: int = 4,
    normalise: bool = False
) -> tf.data.Dataset:
    """
    Create a strictly class-balanced training dataset.

    Uses batch-level balancing: batches half from each class, then concatenates.
    This guarantees exactly 50% class balance in every batch, stabilising
    BatchNorm statistics and gradient updates.

    Args:
        chunk_files: List of chunk file paths
        labels: List of labels for each chunk (0=normal, 1=tumor)
        batch_size: Batch size (will be rounded down to even number)
        max_patches_per_chunk: Memory control (required for safe operation)
        cycle_length: Chunks to read in parallel
        normalise: If True, apply per-patch channel normalisation

    Returns:
        tf.data.Dataset yielding (batch_x, batch_y) with exact 50/50 class balance
    """
    # Split chunks by class
    normal_files = [f for f, l in zip(chunk_files, labels) if l == 0]
    tumor_files = [f for f, l in zip(chunk_files, labels) if l == 1]

    print(f"  Training chunks: {len(normal_files)} normal, {len(tumor_files)} tumor")

    if len(normal_files) == 0 or len(tumor_files) == 0:
        raise ValueError("Need at least one chunk per class for balanced training")

    # Create separate datasets for each class
    normal_ds = _create_single_class_dataset(
        normal_files, label=0,
        max_patches_per_chunk=max_patches_per_chunk,
        cycle_length=max(1, cycle_length // 2)
    )
    tumor_ds = _create_single_class_dataset(
        tumor_files, label=1,
        max_patches_per_chunk=max_patches_per_chunk,
        cycle_length=max(1, cycle_length // 2)
    )

    # Batch each class separately (half the batch size each)
    half_batch = batch_size // 2
    normal_batched = normal_ds.batch(half_batch)
    tumor_batched = tumor_ds.batch(half_batch)

    # Zip and concatenate to form strictly balanced batches
    dataset = tf.data.Dataset.zip((normal_batched, tumor_batched))
    dataset = dataset.map(
        lambda n, t: (
            tf.concat([n[0], t[0]], axis=0),  # Combine patches
            tf.concat([n[1], t[1]], axis=0)   # Combine labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle within batch to mix classes (prevents model learning position)
    dataset = dataset.map(
        lambda x, y: _shuffle_batch(x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Apply per-patch normalisation if enabled
    if normalise:
        dataset = dataset.map(
            lambda x, y: (tf.map_fn(_normalise_patch_channels, x), y),
            num_parallel_calls=2  # Fixed parallelism, not AUTOTUNE
        )

    # Prefetch for performance
    dataset = dataset.prefetch(2)

    return dataset


def create_val_dataset(
    chunk_files: List[str],
    labels: List[int],
    batch_size: int = 32,
    max_patches_per_chunk: int = None
) -> tf.data.Dataset:
    """
    Create a validation dataset using generator for memory safety.

    Uses from_generator to avoid tensor materialization, preventing RAM crashes
    when loading large validation sets without patch limits.

    Args:
        chunk_files: List of chunk file paths
        labels: List of labels for each chunk
        batch_size: Batch size
        max_patches_per_chunk: Memory control (can be None for full validation)

    Returns:
        tf.data.Dataset yielding (batch_x, batch_y)
    """
    # Create generator function (closure to capture parameters)
    def gen():
        return create_chunk_generator(
            chunk_files=chunk_files,
            labels=labels,
            shuffle=False,  # Deterministic validation
            max_patches_per_chunk=max_patches_per_chunk,
            seed=42
        )

    # Create dataset from generator - avoids tensor materialization
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    # Shuffle to mix patches across chunks - prevents class-skewed batches
    # which cause unstable batch norm statistics and jumpy validation metrics.
    # Buffer of 2000 ensures mixing across ~5 chunks (at 400 patches/chunk).
    # reshuffle_each_iteration=False keeps same order each epoch for consistency.
    dataset = dataset.shuffle(buffer_size=2000, seed=42, reshuffle_each_iteration=False)

    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(2)

    # Force determinism
    opts = tf.data.Options()
    opts.deterministic = True
    dataset = dataset.with_options(opts)

    return dataset


def create_preloaded_val_dataset(
    chunk_files: List[str],
    labels: List[int],
    batch_size: int = 32,
    max_samples_per_class: int = 15000,
    normalise: bool = False
) -> Tuple[tf.data.Dataset, int]:
    """
    Create a pre-loaded, cached validation dataset for stable metrics.

    Loads validation patches incrementally (memory-safe), balances classes,
    and interleaves at the sample level (N,T,N,T,...) so every batch is
    ~50% balanced. Uses from_tensor_slices() + cache() for identical results
    every epoch.

    Memory budget: ~4GB for 3000 samples per class (6000 total)
    - 6000 × 224 × 224 × 3 × 4 bytes = ~3.4GB

    This solves validation instability caused by:
    - Generator re-instantiation with fresh RNG state
    - Small shuffle buffers not mixing across chunks
    - Fragile while True + break patterns

    Args:
        chunk_files: List of chunk file paths
        labels: List of labels for each chunk (0=normal, 1=tumor)
        batch_size: Batch size
        max_samples_per_class: Maximum samples per class to load (default 15000)
        normalise: If True, apply per-patch channel normalisation

    Returns:
        (tf.data.Dataset, total_samples) - cached dataset and sample count
    """
    print("Loading validation patches with pre-allocation...")

    # Separate chunk files by class
    normal_files = [f for f, l in zip(chunk_files, labels) if l == 0]
    tumor_files = [f for f, l in zip(chunk_files, labels) if l == 1]

    print(f"  Found {len(normal_files)} normal chunks, {len(tumor_files)} tumor chunks")

    if len(normal_files) == 0 or len(tumor_files) == 0:
        raise ValueError("Need at least one chunk per class")

    def load_class_preallocated(chunk_files_list, max_samples, class_name):
        """Load patches with pre-allocation - no memory spike."""
        patches_per_chunk = max(1, max_samples // len(chunk_files_list))

        # Pre-allocate final array (key fix - avoids append+concatenate spike)
        X = np.empty((max_samples, 224, 224, 3), dtype=np.float32)
        offset = 0

        for i, path in enumerate(chunk_files_list):
            if offset >= max_samples:
                break

            try:
                with np.load(path, mmap_mode='r') as data:
                    X_mmap = data['X']
                    n_to_load = min(patches_per_chunk, len(X_mmap), max_samples - offset)

                    # Load chunk data
                    chunk_data = X_mmap[:n_to_load].astype(np.float32)

                    # Normalise if needed
                    if chunk_data.max() > 1.5:
                        chunk_data = chunk_data / 255.0
                    chunk_data = np.clip(chunk_data, 0.0, 1.0)

                    # Load directly into pre-allocated array (no extra copy)
                    X[offset:offset + n_to_load] = chunk_data
                    offset += n_to_load

                    del chunk_data

            except Exception as e:
                print(f"    Error loading {path}: {e}")
                continue

            gc.collect()

            if (i + 1) % 5 == 0 or (i + 1) == len(chunk_files_list):
                print(f"    {class_name}: chunk {i + 1}/{len(chunk_files_list)}, "
                      f"{offset} patches loaded")

        return X[:offset]  # Trim to actual size

    # Load with pre-allocation (memory-safe)
    all_normal = load_class_preallocated(normal_files, max_samples_per_class, "normal")
    all_tumor = load_class_preallocated(tumor_files, max_samples_per_class, "tumor")

    print(f"  Loaded: {len(all_normal)} normal, {len(all_tumor)} tumor patches")

    # Balance classes: take min(n_normal, n_tumor) from each
    n_per_class = min(len(all_normal), len(all_tumor))

    if n_per_class == 0:
        raise ValueError("No patches found for one or both classes")

    all_normal = all_normal[:n_per_class]
    all_tumor = all_tumor[:n_per_class]

    print(f"  Balanced: {n_per_class} samples per class ({2 * n_per_class} total)")

    # Interleave at sample level: N,T,N,T,N,T...
    # This ensures every batch has ~50% class balance
    interleaved_X = np.empty((2 * n_per_class, 224, 224, 3), dtype=np.float32)
    interleaved_y = np.empty(2 * n_per_class, dtype=np.int32)

    interleaved_X[0::2] = all_normal  # Even indices: normal
    interleaved_X[1::2] = all_tumor   # Odd indices: tumor
    interleaved_y[0::2] = 0
    interleaved_y[1::2] = 1

    del all_normal, all_tumor
    gc.collect()

    # Create dataset with from_tensor_slices (deterministic)
    dataset = tf.data.Dataset.from_tensor_slices((interleaved_X, interleaved_y))

    # Cache for identical results every epoch - NO shuffle, NO repeat
    dataset = dataset.cache()

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Apply per-patch normalisation if enabled
    if normalise:
        dataset = dataset.map(
            lambda x, y: (tf.map_fn(_normalise_patch_channels, x), y),
            num_parallel_calls=2  # Fixed parallelism, not AUTOTUNE
        )

    dataset = dataset.prefetch(2)

    # Force determinism
    opts = tf.data.Options()
    opts.deterministic = True
    dataset = dataset.with_options(opts)

    total_samples = 2 * n_per_class
    print(f"  Created cached validation dataset: {total_samples} samples")

    return dataset, total_samples


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

    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_path}"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{experiment_name.replace(' ', '_')}_"))

    (temp_dir / 'normal').mkdir()
    (temp_dir / 'tumor').mkdir()

    total_linked = 0
    missing_dirs = []

    for binary_label, source_classes in class_mapping.items():
        target_dir = temp_dir / ('normal' if binary_label == 0 else 'tumor')

        if isinstance(source_classes, str):
            source_classes = [source_classes]

        for source_class in source_classes:
            source_dir = source_path / source_class
            if not source_dir.exists():
                missing_dirs.append(str(source_dir))
                continue

            linked = 0
            for chunk_file in source_dir.glob('*.npz'):
                safe_name = f"{source_class}_{chunk_file.name}"
                link_path = target_dir / safe_name

                try:
                    os.symlink(str(chunk_file), str(link_path))
                    linked += 1
                except OSError:
                    try:
                        shutil.copy2(str(chunk_file), str(link_path))
                        linked += 1
                    except Exception as e:
                        print(f"  Warning: failed to link/copy {chunk_file.name}: {e}")

            total_linked += linked

    if total_linked == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)

        available = [d.name for d in source_path.iterdir() if d.is_dir()]
        requested = [
            cls for classes in class_mapping.values()
            for cls in (classes if isinstance(classes, list) else [classes])
        ]
        raise ValueError(
            f"No .npz chunks found for experiment '{experiment_name}'.\n"
            f"  Dataset path: {dataset_path}\n"
            f"  Requested classes: {requested}\n"
            f"  Missing directories: {missing_dirs or 'none (dirs exist but contain no .npz files)'}\n"
            f"  Available subdirectories: {available}"
        )

    return str(temp_dir)


def setup_training_pipeline(
    base_path: str,
    config: TrainingConfig = None,
    use_preloaded_val: bool = True,
    val_max_samples_per_class: int = 15000
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Set up complete training and validation pipelines.

    Args:
        base_path: Path to dataset (binary structure)
        config: Training configuration
        use_preloaded_val: If True, load all validation patches into memory
            with class balancing and interleaving for stable metrics.
            If False, use generator-based approach (legacy behaviour).
        val_max_samples_per_class: Maximum validation samples per class when
            use_preloaded_val=True (default 15000). Set to -1 for no limit.

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
    
    # Create datasets using hybrid approach:
    # - Training: interleave for performance + diversity
    # - Validation: generator for memory safety
    train_dataset = create_train_dataset(
        train_files, train_labels,
        batch_size=config.batch_size,
        max_patches_per_chunk=config.max_patches_per_chunk,
        cycle_length=config.cycle_length,
        normalise=config.normalise_patches
    )
    train_dataset = train_dataset.repeat()

    # Create validation dataset using selected approach
    if use_preloaded_val:
        # Pre-loaded approach: stable, cached, class-balanced
        val_dataset, val_samples = create_preloaded_val_dataset(
            val_files, val_labels,
            batch_size=config.batch_size,
            max_samples_per_class=val_max_samples_per_class,
            normalise=config.normalise_patches
        )
        val_steps = (val_samples + config.batch_size - 1) // config.batch_size
    else:
        # Legacy generator approach (for rollback if needed)
        val_dataset = create_val_dataset(
            val_files, val_labels,
            batch_size=config.batch_size,
            max_patches_per_chunk=None  # Load all validation patches
        )
        val_steps = None  # Let Keras consume full dataset

    # Estimate training steps
    # Rough estimate based on chunk count and max patches
    patches_per_chunk = config.max_patches_per_chunk or 1000
    train_steps = max(100, len(train_files) * patches_per_chunk // config.batch_size)

    print(f"Steps: {train_steps} train, {val_steps or 'auto'} val")

    return train_dataset, val_dataset, train_steps, val_steps


def diagnose_validation_stability(
    model,
    val_dataset: tf.data.Dataset,
    num_checks: int = 5,
    val_steps: int = None
) -> Dict[str, float]:
    """
    Diagnose validation stability by running evaluation multiple times.

    Runs model.evaluate() multiple times on the same dataset and reports
    accuracy variance. Useful for detecting instability caused by:
    - Generator re-instantiation
    - Non-deterministic data loading
    - Batch normalisation sensitivity to batch composition

    Args:
        model: Compiled Keras model
        val_dataset: Validation dataset to test
        num_checks: Number of evaluation runs (default 5)
        val_steps: Number of validation steps (None for auto)

    Returns:
        Dictionary with 'accuracies', 'mean', 'std', 'variance', 'is_stable'
    """
    print(f"\nDiagnosing validation stability ({num_checks} runs)...")

    accuracies = []
    for i in range(num_checks):
        results = model.evaluate(val_dataset, steps=val_steps, verbose=0)

        # Handle both single metric and list of metrics
        if isinstance(results, list):
            # Assume accuracy is second metric (after loss)
            acc = results[1] if len(results) > 1 else results[0]
        else:
            acc = results

        accuracies.append(acc * 100)  # Convert to percentage
        print(f"  Run {i + 1}: {acc * 100:.2f}%")

    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    variance = np.var(accuracies)

    print(f"\nResults:")
    print(f"  Mean accuracy: {mean_acc:.2f}%")
    print(f"  Std deviation: {std_acc:.4f}%")
    print(f"  Variance: {variance:.6f}")

    is_stable = variance < 1.0  # 1% variance threshold

    if is_stable:
        print(f"  Status: STABLE (variance {variance:.4f}% < 1%)")
    else:
        print(f"  WARNING: UNSTABLE (variance {variance:.4f}% > 1%)")
        print(f"  Consider using use_preloaded_val=True for stable validation")

    return {
        'accuracies': accuracies.tolist(),
        'mean': mean_acc,
        'std': std_acc,
        'variance': variance,
        'is_stable': is_stable
    }
