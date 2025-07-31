"""
Configuration file for SmartRetail hybrid recommendation system.
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

# Collaborative Filtering Configuration
COLLABORATIVE_CONFIG = {
    "user_based": {
        "n_neighbors": 50,
        "similarity_metric": "cosine",
        "min_similarity": 0.1,
        "min_common_items": 5
    },
    "item_based": {
        "n_neighbors": 100,
        "similarity_metric": "cosine",
        "min_similarity": 0.1,
        "min_common_users": 3
    },
    "matrix_factorization": {
        "n_factors": 100,
        "n_epochs": 20,
        "learning_rate": 0.01,
        "regularization": 0.1,
        "random_state": 42
    },
    "svd": {
        "n_components": 50,
        "random_state": 42
    },
    "nmf": {
        "n_components": 50,
        "random_state": 42,
        "max_iter": 200
    }
}

# Content-Based Filtering Configuration
CONTENT_BASED_CONFIG = {
    "tfidf": {
        "max_features": 1000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.8
    },
    "word_embeddings": {
        "embedding_dim": 128,
        "window_size": 5,
        "min_count": 2,
        "sg": 1,  # Skip-gram
        "epochs": 10
    },
    "product_embeddings": {
        "embedding_dim": 64,
        "hidden_layers": [128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32
    },
    "similarity": {
        "metric": "cosine",
        "min_similarity": 0.1
    }
}

# Hybrid System Configuration
HYBRID_CONFIG = {
    "weighted": {
        "collaborative_weight": 0.6,
        "content_weight": 0.4,
        "dynamic_weighting": True
    },
    "switching": {
        "confidence_threshold": 0.7,
        "fallback_method": "collaborative"
    },
    "cascade": {
        "primary_method": "collaborative",
        "secondary_method": "content",
        "min_recommendations": 5
    },
    "feature_combination": {
        "collaborative_features": 0.5,
        "content_features": 0.5,
        "fusion_method": "concatenate"
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": ["precision", "recall", "ndcg", "diversity", "coverage", "serendipity"],
    "k_values": [5, 10, 20],
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# Data Processing Configuration
DATA_CONFIG = {
    "min_ratings_per_user": 5,
    "min_ratings_per_item": 3,
    "rating_threshold": 3.5,  # Consider positive ratings above this threshold
    "text_columns": ["title", "description", "category", "brand"],
    "categorical_columns": ["category", "brand", "color", "size"],
    "numerical_columns": ["price", "rating", "review_count"]
}

# Model Persistence Configuration
MODEL_CONFIG = {
    "save_models": True,
    "model_format": "joblib",
    "save_embeddings": True,
    "save_similarity_matrices": True,
    "compression": True
}

# User Profile Configuration
USER_PROFILE_CONFIG = {
    "profile_types": {
        "new_user": {
            "min_interactions": 0,
            "max_interactions": 5,
            "primary_method": "content",
            "exploration_factor": 0.3
        },
        "active_user": {
            "min_interactions": 6,
            "max_interactions": 50,
            "primary_method": "hybrid",
            "exploration_factor": 0.1
        },
        "power_user": {
            "min_interactions": 51,
            "max_interactions": float('inf'),
            "primary_method": "collaborative",
            "exploration_factor": 0.05
        }
    },
    "cold_start_strategy": "content_based",
    "exploration_strategy": "epsilon_greedy",
    "epsilon": 0.1
}

# Feedback Integration Configuration
FEEDBACK_CONFIG = {
    "feedback_types": ["explicit", "implicit", "contextual"],
    "learning_rate": 0.01,
    "update_frequency": "batch",
    "batch_size": 1000,
    "retrain_threshold": 0.1,  # Retrain when performance drops by 10%
    "feedback_weight": 0.3
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "parallel_processing": True,
    "n_jobs": -1,
    "chunk_size": 1000,
    "memory_efficient": True,
    "cache_similarity_matrices": True
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "plot_style": "seaborn-v0_8",
    "figure_size": (12, 8),
    "dpi": 300,
    "color_palette": "husl",
    "interactive_plots": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": True,
    "console_handler": True
}

# Default Parameters for Quick Start
DEFAULT_CONFIG = {
    "n_recommendations": 10,
    "similarity_metric": "cosine",
    "min_similarity": 0.1,
    "random_state": 42,
    "verbose": True
} 