"""
Configuration file for SmartRetail multimodal emotion analysis project.
Contains all hyperparameters, paths, and settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Facial Analysis Configuration (FER2013)
FACIAL_CONFIG = {
    "image_size": (48, 48),
    "num_classes": 7,
    "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "data_augmentation": True,
    "model_save_path": str(MODELS_DIR / "facial_cnn.h5"),
    "history_save_path": str(RESULTS_DIR / "facial_training_history.json")
}

# Text Analysis Configuration (EmoReact)
TEXT_CONFIG = {
    "max_sequence_length": 100,
    "vocab_size": 10000,
    "embedding_dim": 128,
    "num_classes": 6,
    "emotions": ["joy", "sadness", "anger", "fear", "surprise", "neutral"],
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "model_save_path": str(MODELS_DIR / "text_rnn.h5"),
    "history_save_path": str(RESULTS_DIR / "text_training_history.json"),
    "tokenizer_save_path": str(MODELS_DIR / "tokenizer.pkl")
}

# Multimodal Configuration
MULTIMODAL_CONFIG = {
    "fusion_method": "concatenate",  # "concatenate", "attention", "weighted"
    "hidden_dim": 256,
    "dropout_rate": 0.3,
    "batch_size": 16,
    "epochs": 40,
    "learning_rate": 0.0005,
    "validation_split": 0.2,
    "model_save_path": str(MODELS_DIR / "multimodal_model.h5"),
    "history_save_path": str(RESULTS_DIR / "multimodal_training_history.json")
}

# Training Configuration
TRAINING_CONFIG = {
    "random_seed": 42,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1_score"],
    "confusion_matrix_save_path": str(RESULTS_DIR / "confusion_matrix.png"),
    "roc_curve_save_path": str(RESULTS_DIR / "roc_curves.png"),
    "classification_report_save_path": str(RESULTS_DIR / "classification_report.txt")
}

# Data Augmentation for Facial Images
FACIAL_AUGMENTATION = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "horizontal_flip": True,
    "zoom_range": 0.2,
    "fill_mode": "nearest"
}

# Model Architecture Configurations
CNN_ARCHITECTURE = {
    "conv_layers": [
        {"filters": 32, "kernel_size": 3, "activation": "relu"},
        {"filters": 64, "kernel_size": 3, "activation": "relu"},
        {"filters": 128, "kernel_size": 3, "activation": "relu"}
    ],
    "pooling_layers": [
        {"pool_size": 2, "strides": 2},
        {"pool_size": 2, "strides": 2},
        {"pool_size": 2, "strides": 2}
    ],
    "dense_layers": [
        {"units": 256, "activation": "relu", "dropout": 0.5},
        {"units": 128, "activation": "relu", "dropout": 0.3}
    ]
}

RNN_ARCHITECTURE = {
    "lstm_units": [128, 64],
    "bidirectional": True,
    "attention": True,
    "dense_layers": [
        {"units": 128, "activation": "relu", "dropout": 0.3},
        {"units": 64, "activation": "relu", "dropout": 0.2}
    ]
}

TRANSFORMER_ARCHITECTURE = {
    "num_heads": 8,
    "ff_dim": 256,
    "num_transformer_blocks": 4,
    "mlp_units": [128, 64],
    "dropout_rate": 0.1
} 