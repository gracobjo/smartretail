"""
Fraud detection models using supervised learning algorithms.
Implements XGBoost, LightGBM, Random Forest, and Logistic Regression.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

class FraudDetector:
    """Main fraud detection model with multiple algorithms."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.class_weights = None
        
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model for fraud detection.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            xgb.XGBClassifier: Trained XGBoost model
        """
        print("Training XGBoost model...")
        
        # Get XGBoost parameters
        params = self.config['xgboost'].copy()
        
        # Calculate class weights for imbalanced data
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts else 1
        params['scale_pos_weight'] = scale_pos_weight
        
        # Initialize model
        model = xgb.XGBClassifier(**params)
        
        # Train model
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        print("XGBoost training completed!")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train LightGBM model for fraud detection.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            lgb.LGBMClassifier: Trained LightGBM model
        """
        print("Training LightGBM model...")
        
        # Get LightGBM parameters
        params = self.config['lightgbm'].copy()
        
        # Calculate class weights for imbalanced data
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts else 1
        params['scale_pos_weight'] = scale_pos_weight
        
        # Initialize model
        model = lgb.LGBMClassifier(**params)
        
        # Train model
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        print("LightGBM training completed!")
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model for fraud detection.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            RandomForestClassifier: Trained Random Forest model
        """
        print("Training Random Forest model...")
        
        # Get Random Forest parameters
        params = self.config['random_forest'].copy()
        
        # Calculate class weights for imbalanced data
        class_counts = y_train.value_counts()
        class_weight = {0: 1, 1: class_counts[0] / class_counts[1]} if 1 in class_counts else 'balanced'
        params['class_weight'] = class_weight
        
        # Initialize model
        model = RandomForestClassifier(**params)
        
        # Train model
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        print("Random Forest training completed!")
        
        return model
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression model for fraud detection.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            LogisticRegression: Trained Logistic Regression model
        """
        print("Training Logistic Regression model...")
        
        # Get Logistic Regression parameters
        params = self.config['logistic_regression'].copy()
        
        # Calculate class weights for imbalanced data
        class_counts = y_train.value_counts()
        class_weight = {0: 1, 1: class_counts[0] / class_counts[1]} if 1 in class_counts else 'balanced'
        params['class_weight'] = class_weight
        
        # Initialize model
        model = LogisticRegression(**params)
        
        # Train model
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        print("Logistic Regression training completed!")
        
        return model
    
    def balance_data(self, X, y, method='smote'):
        """
        Balance imbalanced dataset.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            method (str): Balancing method
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        print(f"Balancing data using {method}...")
        
        if method == 'smote':
            balancer = SMOTE(random_state=self.config['random_state'])
        elif method == 'random_oversampling':
            from imblearn.over_sampling import RandomOverSampler
            balancer = RandomOverSampler(random_state=self.config['random_state'])
        elif method == 'random_undersampling':
            balancer = RandomUnderSampler(random_state=self.config['random_state'])
        elif method == 'smoteenn':
            balancer = SMOTEENN(random_state=self.config['random_state'])
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        
        print(f"Original distribution: {y.value_counts().to_dict()}")
        print(f"Balanced distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, balance_data=True):
        """
        Train all available models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            balance_data (bool): Whether to balance the data
            
        Returns:
            dict: Trained models
        """
        print("Training all models...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Balance data if requested
        if balance_data:
            X_train_balanced, y_train_balanced = self.balance_data(
                X_train, y_train, method=self.config['method']
            )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train XGBoost
        try:
            self.train_xgboost(X_train_balanced, y_train_balanced, X_val, y_val)
        except Exception as e:
            print(f"XGBoost training failed: {e}")
        
        # Train LightGBM
        try:
            self.train_lightgbm(X_train_balanced, y_train_balanced, X_val, y_val)
        except Exception as e:
            print(f"LightGBM training failed: {e}")
        
        # Train Random Forest
        try:
            self.train_random_forest(X_train_balanced, y_train_balanced)
        except Exception as e:
            print(f"Random Forest training failed: {e}")
        
        # Train Logistic Regression
        try:
            self.train_logistic_regression(X_train_balanced, y_train_balanced)
        except Exception as e:
            print(f"Logistic Regression training failed: {e}")
        
        print(f"All models trained successfully! Available models: {list(self.models.keys())}")
        return self.models
    
    def predict(self, X, model_name=None):
        """
        Make predictions using trained models.
        
        Args:
            X (pd.DataFrame): Features to predict
            model_name (str): Specific model to use
            
        Returns:
            dict: Predictions from all models or specific model
        """
        if not self.models:
            raise ValueError("No models trained. Call train_all_models() first.")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
            
            model = self.models[model_name]
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'model': model_name
            }
        else:
            results = {}
            for name, model in self.models.items():
                try:
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1]
                    results[name] = {
                        'predictions': predictions,
                        'probabilities': probabilities
                    }
                except Exception as e:
                    print(f"Prediction failed for {name}: {e}")
            
            return results
    
    def get_feature_importance(self, model_name=None):
        """
        Get feature importance from trained models.
        
        Args:
            model_name (str): Specific model to use
            
        Returns:
            dict: Feature importance for all models or specific model
        """
        if not self.models:
            raise ValueError("No models trained. Call train_all_models() first.")
        
        if not self.feature_names:
            raise ValueError("Feature names not available.")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
            
            model = self.models[model_name]
            importance = self._extract_feature_importance(model, model_name)
            
            return {
                'feature_names': self.feature_names,
                'importance': importance,
                'model': model_name
            }
        else:
            results = {}
            for name, model in self.models.items():
                try:
                    importance = self._extract_feature_importance(model, name)
                    results[name] = {
                        'feature_names': self.feature_names,
                        'importance': importance
                    }
                except Exception as e:
                    print(f"Feature importance extraction failed for {name}: {e}")
            
            return results
    
    def _extract_feature_importance(self, model, model_name):
        """
        Extract feature importance from a model.
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            
        Returns:
            np.array: Feature importance values
        """
        if model_name in ['xgboost', 'lightgbm']:
            return model.feature_importances_
        elif model_name == 'random_forest':
            return model.feature_importances_
        elif model_name == 'logistic_regression':
            return np.abs(model.coef_[0])
        else:
            return np.zeros(len(self.feature_names))
    
    def cross_validate(self, X, y, cv_folds=5):
        """
        Perform cross-validation on all models.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Cross-validation results
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config['random_state'])
        results = {}
        
        for name, model in self.models.items():
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
                
                results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
                
                print(f"{name}: F1 = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Cross-validation failed for {name}: {e}")
        
        return results
    
    def select_best_model(self, X_val, y_val, metric='f1'):
        """
        Select the best model based on validation performance.
        
        Args:
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            metric (str): Metric to use for selection
            
        Returns:
            str: Name of the best model
        """
        print("Selecting best model...")
        
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        
        best_score = 0
        best_model = None
        
        for name, model in self.models.items():
            try:
                predictions = model.predict(X_val)
                probabilities = model.predict_proba(X_val)[:, 1]
                
                if metric == 'f1':
                    score = f1_score(y_val, predictions)
                elif metric == 'precision':
                    score = precision_score(y_val, predictions)
                elif metric == 'recall':
                    score = recall_score(y_val, predictions)
                elif metric == 'roc_auc':
                    score = roc_auc_score(y_val, probabilities)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                print(f"{name}: {metric} = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = name
                    
            except Exception as e:
                print(f"Evaluation failed for {name}: {e}")
        
        if best_model:
            self.best_model = best_model
            print(f"Best model: {best_model} ({metric} = {best_score:.4f})")
        else:
            print("No best model selected.")
        
        return best_model
    
    def save_models(self, filepath):
        """
        Save trained models to disk.
        
        Args:
            filepath (str): Path to save models
        """
        import joblib
        
        print(f"Saving models to {filepath}...")
        
        model_data = {
            'models': self.models,
            'feature_names': self.feature_names,
            'best_model': self.best_model,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        print("Models saved successfully!")
    
    def load_models(self, filepath):
        """
        Load trained models from disk.
        
        Args:
            filepath (str): Path to load models from
        """
        import joblib
        
        print(f"Loading models from {filepath}...")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_names = model_data['feature_names']
        self.best_model = model_data['best_model']
        
        print("Models loaded successfully!")
        print(f"Available models: {list(self.models.keys())}") 