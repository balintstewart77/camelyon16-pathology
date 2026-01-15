"""
Training utilities for binary classification experiments.

Provides a clean interface for running experiments with different
class combinations from the 4-class dataset.
"""

import gc
import shutil
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
    create_binary_dataset
)
from src.models.architectures import get_model


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
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
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


def evaluate_model(
    model: keras.Model,
    val_dataset: tf.data.Dataset,
    experiment_name: str,
    history: keras.callbacks.History = None,
    threshold: float = 0.5,
    keep_predictions: bool = False
) -> Dict:
    """
    Comprehensive model evaluation with visualizations.

    Args:
        model: Trained Keras model
        val_dataset: Validation dataset
        experiment_name: Name for plots
        history: Training history (optional)
        threshold: Classification threshold
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
    fig.suptitle(f'{experiment_name}\nAccuracy: {accuracy:.1%}, AUC: {auc:.3f}')
    
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
    axes[1, 0].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
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


def run_binary_experiment(
    dataset_path: str,
    experiment_type: int,
    model_name: str = 'simple',
    config: TrainingConfig = None,
    epochs: int = 20,
    learning_rate: float = 1e-4,
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
        train_ds, val_ds, train_steps, _ = setup_training_pipeline(binary_path, config)
        
        # Build model
        model = get_model(model_name)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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
