"""
Configuration settings for the deep learning classifier project.
"""

# General configuration
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
VALIDATION_SPLIT = 0.2
MODEL_SAVE_PATH = "saved_models"

# Image classification configuration
IMAGE_CONFIG = {
    "cifar10": {
        "img_size": (32, 32),
        "num_classes": 10,
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
    },
    "mnist": {
        "img_size": (28, 28),
        "num_classes": 10,
        "class_names": [str(i) for i in range(10)],
    },
}

# Data augmentation parameters
IMAGE_AUGMENTATION = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "zoom_range": 0.1,
}

# CNN architecture parameters
CNN_CONFIG = {
    "filters": [32, 64, 128],
    "kernel_size": (3, 3),
    "pool_size": (2, 2),
    "dropout_rate": 0.5,
    "dense_units": [256],
}

# Text classification configuration
TEXT_CONFIG = {
    "imdb": {
        "max_length": 256,
        "num_classes": 2,
        "class_names": ["negative", "positive"],
    },
}

# Transformer model parameters
TRANSFORMER_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 256,
    "dropout_rate": 0.1,
}

# Grad-CAM configuration
GRADCAM_CONFIG = {
    "layer_name": "conv2d_2",  # Target layer for visualization
    "alpha": 0.5,  # Heatmap transparency
}

# SHAP configuration
SHAP_CONFIG = {
    "num_samples": 100,  # Number of samples for SHAP analysis
    "top_n_tokens": 10,  # Number of tokens to show in explanation
}
