"""
Configuration module for fraud detection system.
Contains all parameters and settings for the system.
"""

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'stratify': True,
    'shuffle': True,
    
    # XGBoost parameters
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 10
    },
    
    # LightGBM parameters
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt'
    },
    
    # Random Forest parameters
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    },
    
    # Logistic Regression parameters
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'random_state': 42,
        'max_iter': 1000
    },
    
    # Data balancing method
    'method': 'smote',  # Options: 'smote', 'random_oversampling', 'random_undersampling', 'smoteenn'
    
    # SHAP configuration
    'background_samples': 100,
    'max_display': 20,
    'plot_type': 'bar',
    'feature_names': None,
    
    # Feature engineering
    'feature_engineering': {
        'create_polynomial_features': True,
        'create_interaction_features': True,
        'create_ratio_features': True,
        'create_categorical_features': True,
        'polynomial_degree': 2
    },
    
    # Data processing
    'data_processing': {
        'handle_missing_values': True,
        'handle_outliers': True,
        'outlier_method': 'iqr',  # Options: 'iqr', 'zscore'
        'scale_features': True,
        'encoding_method': 'label'  # Options: 'label', 'onehot'
    },
    
    # Evaluation metrics
    'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'cv_folds': 5,
    
    # Threshold optimization
    'threshold_optimization': {
        'metric': 'f1',
        'threshold_range': (0.1, 0.9),
        'threshold_step': 0.05
    },
    
    # Hyperparameter tuning
    'hyperparameter_tuning': {
        'n_trials': 50,
        'cv_folds': 5,
        'optimization_metric': 'f1'
    },
    
    # File paths
    'paths': {
        'data_dir': 'fraud_detection/data/',
        'models_dir': 'fraud_detection/models/',
        'results_dir': 'fraud_detection/results/',
        'logs_dir': 'fraud_detection/logs/'
    },
    
    # Logging
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'fraud_detection/logs/fraud_detection.log'
    }
}

# Feature selection configuration
FEATURE_SELECTION_CONFIG = {
    'correlation_threshold': 0.95,
    'variance_threshold': 0.01,
    'mutual_info_threshold': 0.01,
    'recursive_feature_elimination': {
        'n_features_to_select': 0.5,
        'step': 0.1
    },
    'shap_based_selection': {
        'importance_threshold': 0.01,
        'top_n_features': 20
    }
}

# Model comparison configuration
MODEL_COMPARISON_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision'],
    'statistical_tests': ['wilcoxon', 'mann_whitney'],
    'confidence_level': 0.95,
    'bootstrap_samples': 1000
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8',
    'palette': 'Set2',
    'figure_size': (12, 8),
    'dpi': 300,
    'save_format': 'png',
    'show_plots': True
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'excellent': {
        'precision': 0.95,
        'recall': 0.90,
        'f1': 0.92,
        'roc_auc': 0.95
    },
    'good': {
        'precision': 0.85,
        'recall': 0.80,
        'f1': 0.82,
        'roc_auc': 0.90
    },
    'acceptable': {
        'precision': 0.75,
        'recall': 0.70,
        'f1': 0.72,
        'roc_auc': 0.80
    }
}

# Fraud detection specific configuration
FRAUD_DETECTION_CONFIG = {
    'fraud_indicators': {
        'high_amount_threshold': 0.95,  # 95th percentile
        'suspicious_time_windows': [(0, 6), (22, 24)],  # Night hours
        'unusual_frequency_threshold': 3,  # Transactions per hour
        'geographic_anomaly_threshold': 0.1  # Distance from usual location
    },
    
    'risk_scoring': {
        'high_risk_threshold': 0.8,
        'medium_risk_threshold': 0.5,
        'low_risk_threshold': 0.2
    },
    
    'alert_system': {
        'immediate_action_threshold': 0.9,
        'review_threshold': 0.7,
        'monitoring_threshold': 0.5
    }
}

# Data validation rules
DATA_VALIDATION_RULES = {
    'required_columns': ['amount', 'time', 'location'],
    'data_types': {
        'amount': 'float64',
        'time': 'int64',
        'location': 'int64',
        'fraud': 'int64'
    },
    'value_ranges': {
        'amount': (0, float('inf')),
        'time': (0, 24),
        'location': (0, 100),
        'fraud': (0, 1)
    },
    'missing_value_threshold': 0.1  # Maximum 10% missing values
}

# Export all configurations
__all__ = [
    'MODEL_CONFIG',
    'FEATURE_SELECTION_CONFIG', 
    'MODEL_COMPARISON_CONFIG',
    'VISUALIZATION_CONFIG',
    'PERFORMANCE_THRESHOLDS',
    'FRAUD_DETECTION_CONFIG',
    'DATA_VALIDATION_RULES'
] 