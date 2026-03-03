"""
Model architectures and training utilities.
"""

from .architectures import (
    build_simple_cnn,
    build_subtle_model,
    build_attention_model,
    build_transfer_model,
    build_transfer_finetune_model,
    MODEL_REGISTRY
)
from .training import (
    run_binary_experiment,
    evaluate_model,
    evaluate_on_test_set,
    find_optimal_threshold,
    save_model_metadata,
    load_model_metadata
)

__all__ = [
    'build_simple_cnn',
    'build_subtle_model',
    'build_attention_model',
    'build_transfer_model',
    'build_transfer_finetune_model',
    'MODEL_REGISTRY',
    'run_binary_experiment',
    'evaluate_model',
    'evaluate_on_test_set',
    'find_optimal_threshold',
    'save_model_metadata',
    'load_model_metadata'
]
