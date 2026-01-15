"""
CNN architectures for patch classification.

Simple, interpretable architectures that work well for WSI patch classification.
These are intentionally kept simple for educational purposes.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_simple_cnn(input_shape=(224, 224, 3)) -> keras.Model:
    """
    Simple CNN that works well for basic tumor detection.
    
    Architecture:
    - 3 convolutional blocks with increasing filters
    - Global average pooling (reduces overfitting vs flatten)
    - Single dense layer with dropout
    
    This model is good for:
    - Baseline experiments
    - Tasks with clear visual differences (normal vs pure tumor)
    
    Parameters: ~200K
    
    Args:
        input_shape: Input image shape
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: 224 -> 112
    x = layers.Conv2D(16, 5, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2: 112 -> 56
    x = layers.Conv2D(32, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3: 56 -> 28
    x = layers.Conv2D(64, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Global pooling -> prediction
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='simple_cnn')


def build_subtle_model(input_shape=(224, 224, 3)) -> keras.Model:
    """
    Model for detecting subtle contextual differences.
    
    Designed for harder tasks like:
    - Normal from normal vs normal from tumor slides
    - Boundary tissue detection
    
    Key differences from simple CNN:
    - Smaller kernels (3x3) to capture fine details
    - More filters for complex pattern detection
    - Gentler stride in first layer (preserves spatial info)
    
    Parameters: ~1.5M
    
    Args:
        input_shape: Input image shape
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: 224 -> 224 (no downsampling yet - preserve detail)
    x = layers.Conv2D(32, 3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Block 2: 224 -> 112
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3: 112 -> 56
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 4: 56 -> 28
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Global pooling -> prediction
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='subtle_model')


def build_attention_model(input_shape=(224, 224, 3)) -> keras.Model:
    """
    Model with spatial attention mechanism.
    
    The attention layer learns to focus on the most discriminative
    regions of each patch. Useful when the relevant signal is
    localized to specific areas.
    
    Args:
        input_shape: Input image shape
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Feature extraction backbone (same as subtle model)
    x = layers.Conv2D(32, 3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Spatial attention: learn where to focus
    # 1x1 conv reduces channels to 1, sigmoid gives attention weights
    attention = layers.Conv2D(1, 1, activation='sigmoid', name='attention_map')(x)
    x = layers.Multiply()([x, attention])  # Weight features by attention
    
    x = layers.Dropout(0.4)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='attention_model')


# Registry for easy access
MODEL_REGISTRY = {
    'simple': build_simple_cnn,
    'subtle': build_subtle_model,
    'attention': build_attention_model
}


def get_model(name: str, **kwargs) -> keras.Model:
    """
    Get a model by name.
    
    Args:
        name: Model name ('simple', 'subtle', 'attention')
        **kwargs: Arguments passed to model builder
        
    Returns:
        Keras model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[name](**kwargs)
