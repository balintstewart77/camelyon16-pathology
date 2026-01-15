"""
Model architectures and training utilities.
"""

from .architectures import build_simple_cnn, build_subtle_model, MODEL_REGISTRY
from .training import run_binary_experiment, evaluate_model

__all__ = [
    'build_simple_cnn',
    'build_subtle_model', 
    'MODEL_REGISTRY',
    'run_binary_experiment',
    'evaluate_model'
]
