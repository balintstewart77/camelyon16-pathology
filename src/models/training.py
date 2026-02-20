"""
Training utilities for binary classification experiments.

Provides a clean interface for running experiments with different
class combinations from the 4-class dataset.
"""

import gc
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import TrainingConfig, DEFAULT_CONFIG

from src.dataset.tf_pipeline import (
    setup_training_pipeline,
    create_binary_dataset,
    load_chunk_paths
)
from src.models.architectures import get_model


def save_model_metadata(
    model_path: str,
    threshold: float,
    experiment_name: str,
    accuracy: float = None,
    auc: float = None
) -> None:
    """
    Save model metadata (threshold, metrics) to a JSON file.

    The metadata file is saved alongside the model with .json extension.

    Args:
        model_path: Path to the saved .keras model file
        threshold: Optimal classification threshold from validation
        experiment_name: Name of the experiment
        accuracy: Validation accuracy (optional)
        auc: Validation AUC (optional)
    """
    metadata_path = Path(model_path).with_suffix('.json')
    metadata = {
        'threshold': threshold,
        'experiment_name': experiment_name,
        'accuracy': accuracy,
        'auc': auc
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}")


def load_model_metadata(model_path: str) -> Dict:
    """
    Load model metadata from the JSON file alongside the model.

    Args:
        model_path: Path to the .keras model file

    Returns:
        Dict with threshold, experiment_name, and optional metrics
    """
    metadata_path = Path(model_path).with_suffix('.json')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        return json.load(f)


class GarbageCollectorCallback(keras.callbacks.Callback):
    """Free memory after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def create_callbacks(
    experiment_name: str,
    save_dir: str = "./models"
) -> list:
    """
    Create standard training callbacks.
    
    Includes:
    - Early stopping on validation loss
    - Learning rate reduction on plateau
    - Model checkpointing
    - Garbage collection
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f"{save_dir}/{experiment_name.replace(' ', '_').lower()}.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        GarbageCollectorCallback()
    ]


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the optimal classification threshold using Youden's J statistic.

    Maximises J = TPR - FPR (sensitivity + specificity - 1),
    which is the standard approach for binary classification
    with potentially imbalanced classes.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities

    Returns:
        Optimal threshold value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def evaluate_model(
    model: keras.Model,
    val_dataset: tf.data.Dataset,
    experiment_name: str,
    history: keras.callbacks.History = None,
    threshold: float = None,
    keep_predictions: bool = False
) -> Dict:
    """
    Comprehensive model evaluation with visualisations.

    If threshold is None, the optimal threshold is found automatically
    using Youden's J statistic on the validation predictions.

    Args:
        model: Trained Keras model
        val_dataset: Validation dataset
        experiment_name: Name for plots
        history: Training history (optional)
        threshold: Classification threshold (None = find optimal)
        keep_predictions: If False, discard y_true/y_prob/y_pred arrays to save RAM

    Returns:
        Dictionary with metrics (and predictions if keep_predictions=True)
    """
    print(f"\n{'='*50}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*50}")

    # Collect predictions
    y_true, y_prob = [], []

    for batch_x, batch_y in val_dataset:
        probs = model.predict(batch_x, verbose=0)
        y_prob.extend(probs.flatten())
        y_true.extend(batch_y.numpy().flatten())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)
        print(f"Optimal threshold: {threshold:.3f}")
    else:
        print(f"Using fixed threshold: {threshold:.3f}")

    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"Samples: {len(y_true):,}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Tumor']))

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{experiment_name}\nAccuracy: {accuracy:.1%}, AUC: {auc:.3f}, Threshold: {threshold:.3f}')

    # 1. Training curves (if history provided)
    if history:
        axes[0, 0].plot(history.history['loss'], label='Train')
        if 'val_loss' in history.history:
            axes[0, 0].plot(history.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if 'accuracy' in history.history:
            axes[0, 1].plot(history.history['accuracy'], label='Train')
            if 'val_accuracy' in history.history:
                axes[0, 1].plot(history.history['val_accuracy'], label='Val')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No history', ha='center', va='center')
        axes[0, 1].text(0.5, 0.5, 'No history', ha='center', va='center')

    # 2. Prediction distribution
    axes[1, 0].hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Normal', density=True)
    axes[1, 0].hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Tumor', density=True)
    axes[1, 0].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.3f}')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 3. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    im = axes[1, 1].imshow(cm, cmap='Blues')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Normal', 'Tumor'])
    axes[1, 1].set_yticklabels(['Normal', 'Tumor'])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')
    axes[1, 1].set_title('Confusion Matrix')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=14)

    plt.tight_layout()
    plt.show()

    result = {
        'accuracy': accuracy,
        'auc': auc,
        'threshold': threshold,
        'confusion_matrix': cm
    }

    # Only keep large arrays if explicitly requested (saves RAM)
    if keep_predictions:
        result['y_true'] = y_true
        result['y_prob'] = y_prob
        result['y_pred'] = y_pred

    return result


def evaluate_on_test_set(
    model: keras.Model,
    test_dataset_path: str,
    class_mapping: Dict,
    model_name: str,
    threshold: float = 0.5,
    batch_size: int = 64,
    normalise: bool = False
) -> Dict:
    """
    Evaluate a trained model on a held-out test set.

    Uses the threshold found on the validation set (from evaluate_model)
    to avoid data leakage. Do NOT find a new threshold on test data.

    Loads and processes patches in batches to avoid memory issues.

    Args:
        model: Trained Keras model
        test_dataset_path: Path to test dataset (4-class format)
        class_mapping: Dict mapping binary labels to 4-class labels,
                       e.g. {0: ['normal_from_normal'], 1: ['pure_tumor']}
        model_name: Name for display and temp directory naming
        threshold: Classification threshold (use value from validation)
        batch_size: Batch size for prediction
        normalise: Apply per-patch normalisation (zero mean, unit std per channel)

    Returns:
        Dict with accuracy, AUC, classification report, and predictions
    """
    binary_test_path = create_binary_dataset(
        test_dataset_path, class_mapping, f"test_{model_name}"
    )

    try:
        chunks = load_chunk_paths(binary_test_path)

        # Process chunks in batches to avoid loading all into memory
        y_true_all = []
        y_pred_proba_all = []

        print(f"Processing {len(chunks)} chunks...")
        for i, (chunk_path, label) in enumerate(chunks):
            with np.load(chunk_path) as data:
                X_chunk = data['X'].astype(np.float32)  # Ensure float32

                # Apply same preprocessing as training pipeline
                if X_chunk.max() > 1.5:
                    X_chunk = X_chunk / 255.0
                X_chunk = np.clip(X_chunk, 0.0, 1.0)

                # Apply per-patch normalisation if model was trained with it
                # (zero mean, unit std per channel)
                if normalise:
                    mean = X_chunk.mean(axis=(1, 2), keepdims=True)
                    std = X_chunk.std(axis=(1, 2), keepdims=True) + 1e-7
                    X_chunk = (X_chunk - mean) / std

                n_samples = len(X_chunk)
                y_true_all.extend([label] * n_samples)

                # Predict in batches within the chunk
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_proba = model.predict(
                        X_chunk[start:end], verbose=0
                    )
                    y_pred_proba_all.extend(batch_proba.flatten())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks")

        y_test = np.array(y_true_all)
        y_pred_proba = np.array(y_pred_proba_all)
        y_pred = (y_pred_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(
            y_test, y_pred, target_names=['Normal', 'Tumor']
        )

        print(f"Processed {len(y_test):,} samples total")

        return {
            'accuracy': accuracy,
            'auc': auc,
            'report': report,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    finally:
        try:
            shutil.rmtree(binary_test_path)
        except Exception:
            pass


def run_binary_experiment(
    dataset_path: str,
    experiment_type: int,
    model_name: str = 'simple',
    config: TrainingConfig = None,
    epochs: int = 20,
    learning_rate: float = 1e-5,
    keep_predictions: bool = False
) -> Dict:
    """
    Run a binary classification experiment.
    
    Experiment types:
        1: Normal vs Any Tumor (0 vs 1,2,3)
        2: Normal vs Pure Tumor (0 vs 3)
        3: Slide Context (0 vs 1) - can we detect tumor-adjacent tissue?
        4: Normal vs Actual Tumor (0 vs 2,3)
        5: Normal vs Boundary (0 vs 2)
    
    Args:
        dataset_path: Path to 4-class dataset
        experiment_type: Which experiment to run (1-5)
        model_name: Model architecture ('simple', 'subtle', 'attention')
        config: Training configuration
        epochs: Maximum training epochs
        learning_rate: Initial learning rate
        keep_predictions: If False, discard prediction arrays to save RAM

    Returns:
        Dictionary with model, history, and results
    """
    if config is None:
        config = DEFAULT_CONFIG.training
    
    # Define experiments
    experiments = {
        1: {
            'name': 'Normal vs Any Tumor',
            'mapping': {0: ['normal_from_normal'], 1: ['normal_from_tumor', 'boundary_tumor', 'pure_tumor']}
        },
        2: {
            'name': 'Normal vs Pure Tumor',
            'mapping': {0: ['normal_from_normal'], 1: ['pure_tumor']}
        },
        3: {
            'name': 'Slide Context Detection',
            'mapping': {0: ['normal_from_normal'], 1: ['normal_from_tumor']}
        },
        4: {
            'name': 'Normal vs Actual Tumor',
            'mapping': {0: ['normal_from_normal'], 1: ['boundary_tumor', 'pure_tumor']}
        },
        5: {
            'name': 'Normal vs Boundary',
            'mapping': {0: ['normal_from_normal'], 1: ['boundary_tumor']}
        }
    }
    
    if experiment_type not in experiments:
        raise ValueError(f"experiment_type must be 1-5, got {experiment_type}")
    
    exp = experiments[experiment_type]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp['name']}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Mapping: {exp['mapping']}")
    
    # Create binary dataset
    binary_path = create_binary_dataset(dataset_path, exp['mapping'], exp['name'])
    
    try:
        # Setup pipeline
        train_ds, val_ds, train_steps, _ = setup_training_pipeline(
            binary_path,
            config,
            use_preloaded_val=config.use_preloaded_val,
            val_max_samples_per_class=config.val_max_samples_per_class
        )
        
        # Build model
        model = get_model(model_name)
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0 if model_name == 'subtle' else None
        )
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print(f"\nModel: {model.count_params():,} parameters")
        print(f"Training: {train_steps} steps/epoch, {epochs} max epochs")
        
        # Train
        callbacks = create_callbacks(exp['name'])
        
        history = model.fit(
            train_ds,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        results = evaluate_model(model, val_ds, exp['name'], history, keep_predictions=keep_predictions)

        # Save model metadata (threshold, metrics) alongside the model
        model_path = f"./models/{exp['name'].replace(' ', '_').lower()}.keras"
        save_model_metadata(
            model_path,
            threshold=results['threshold'],
            experiment_name=exp['name'],
            accuracy=results['accuracy'],
            auc=results['auc']
        )

        return {
            'model': model,
            'history': history,
            'results': results,
            'experiment_name': exp['name']
        }
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(binary_path)
        except:
            pass
