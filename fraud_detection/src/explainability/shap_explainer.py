"""
SHAP explainability module for fraud detection models.
Provides comprehensive model interpretability using SHAP values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    """SHAP explainer for fraud detection models."""
    
    def __init__(self, config):
        self.config = config
        self.explainers = {}
        self.shap_values = {}
        self.background_data = None
        
    def create_explainer(self, model, X_train, model_name):
        """
        Create SHAP explainer for a model.
        
        Args:
            model: Trained model
            X_train (pd.DataFrame): Training data
            model_name (str): Name of the model
            
        Returns:
            shap.Explainer: SHAP explainer
        """
        print(f"Creating SHAP explainer for {model_name}...")
        
        # Create background data
        if self.config['background_samples'] < len(X_train):
            background_indices = np.random.choice(
                len(X_train), 
                self.config['background_samples'], 
                replace=False
            )
            background_data = X_train.iloc[background_indices]
        else:
            background_data = X_train
        
        self.background_data = background_data
        
        # Create explainer based on model type
        if model_name in ['xgboost', 'lightgbm', 'random_forest']:
            explainer = shap.TreeExplainer(model, background_data)
        elif model_name == 'logistic_regression':
            explainer = shap.LinearExplainer(model, background_data)
        else:
            # Fallback to KernelExplainer
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
        
        self.explainers[model_name] = explainer
        print(f"SHAP explainer created for {model_name}")
        
        return explainer
    
    def calculate_shap_values(self, X, model_name):
        """
        Calculate SHAP values for a dataset.
        
        Args:
            X (pd.DataFrame): Features to explain
            model_name (str): Name of the model
            
        Returns:
            np.array: SHAP values
        """
        if model_name not in self.explainers:
            raise ValueError(f"Explainer not found for {model_name}. Call create_explainer() first.")
        
        print(f"Calculating SHAP values for {model_name}...")
        
        explainer = self.explainers[model_name]
        shap_values = explainer.shap_values(X, check_additivity=False)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # For binary classification, use positive class values
            shap_values = shap_values[1]
        
        self.shap_values[model_name] = shap_values
        print(f"SHAP values calculated for {model_name}")
        
        return shap_values
    
    def plot_summary(self, X, model_name, save_path=None):
        """
        Plot SHAP summary plot.
        
        Args:
            X (pd.DataFrame): Features to explain
            model_name (str): Name of the model
            save_path (str): Path to save plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(X, model_name)
        
        shap_values = self.shap_values[model_name]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X,
            max_display=self.config['max_display'],
            plot_type=self.config['plot_type'],
            show=False
        )
        
        plt.title(f'SHAP Summary Plot - {model_name.upper()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_force_plot(self, X, model_name, sample_idx=0, save_path=None):
        """
        Plot SHAP force plot for a specific sample.
        
        Args:
            X (pd.DataFrame): Features to explain
            model_name (str): Name of the model
            sample_idx (int): Index of sample to explain
            save_path (str): Path to save plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(X, model_name)
        
        shap_values = self.shap_values[model_name]
        
        plt.figure(figsize=(12, 6))
        shap.force_plot(
            self.explainers[model_name].expected_value,
            shap_values[sample_idx],
            X.iloc[sample_idx],
            show=False
        )
        
        plt.title(f'SHAP Force Plot - {model_name.upper()} (Sample {sample_idx})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_waterfall(self, X, model_name, sample_idx=0, save_path=None):
        """
        Plot SHAP waterfall plot for a specific sample.
        
        Args:
            X (pd.DataFrame): Features to explain
            model_name (str): Name of the model
            sample_idx (int): Index of sample to explain
            save_path (str): Path to save plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(X, model_name)
        
        shap_values = self.shap_values[model_name]
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=self.explainers[model_name].expected_value,
                data=X.iloc[sample_idx]
            ),
            show=False
        )
        
        plt.title(f'SHAP Waterfall Plot - {model_name.upper()} (Sample {sample_idx})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_dependence(self, X, model_name, feature_name, save_path=None):
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            X (pd.DataFrame): Features to explain
            model_name (str): Name of the model
            feature_name (str): Name of the feature to plot
            save_path (str): Path to save plot
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(X, model_name)
        
        shap_values = self.shap_values[model_name]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            shap_values,
            X,
            show=False
        )
        
        plt.title(f'SHAP Dependence Plot - {model_name.upper()} ({feature_name})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interaction_values(self, X, model_name, feature1, feature2, save_path=None):
        """
        Plot SHAP interaction values for two features.
        
        Args:
            X (pd.DataFrame): Features to explain
            model_name (str): Name of the model
            feature1 (str): First feature name
            feature2 (str): Second feature name
            save_path (str): Path to save plot
        """
        if model_name not in self.explainers:
            raise ValueError(f"Explainer not found for {model_name}")
        
        print(f"Calculating interaction values for {model_name}...")
        
        explainer = self.explainers[model_name]
        interaction_values = explainer.shap_interaction_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            (feature1, feature2),
            interaction_values,
            X,
            show=False
        )
        
        plt.title(f'SHAP Interaction Plot - {model_name.upper()} ({feature1} vs {feature2})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_feature_importance(self, model_name, top_n=10):
        """
        Get feature importance based on SHAP values.
        
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not found for {model_name}")
        
        shap_values = self.shap_values[model_name]
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dataframe
        # Ensure feature names match the SHAP values length
        if self.config['feature_names'] and len(self.config['feature_names']) == len(mean_shap):
            feature_names = self.config['feature_names']
        else:
            feature_names = [f'feature_{i}' for i in range(len(mean_shap))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def explain_prediction(self, X, model_name, sample_idx=0):
        """
        Explain a specific prediction.
        
        Args:
            X (pd.DataFrame): Features to explain
            model_name (str): Name of the model
            sample_idx (int): Index of sample to explain
            
        Returns:
            dict: Explanation details
        """
        if model_name not in self.shap_values:
            self.calculate_shap_values(X, model_name)
        
        shap_values = self.shap_values[model_name]
        sample_values = shap_values[sample_idx]
        sample_features = X.iloc[sample_idx]
        
        # Get feature names
        feature_names = self.config['feature_names'] if self.config['feature_names'] else [f'feature_{i}' for i in range(len(sample_values))]
        
        # Create explanation dictionary
        explanation = {
            'sample_index': sample_idx,
            'base_value': self.explainers[model_name].expected_value,
            'prediction': sample_values.sum() + self.explainers[model_name].expected_value,
            'feature_contributions': {}
        }
        
        # Add feature contributions
        for i, (feature, value) in enumerate(zip(feature_names, sample_values)):
            explanation['feature_contributions'][feature] = {
                'shap_value': value,
                'feature_value': sample_features.iloc[i] if i < len(sample_features) else 0
            }
        
        # Sort by absolute SHAP value
        sorted_features = sorted(
            explanation['feature_contributions'].items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )
        
        explanation['sorted_features'] = sorted_features
        
        return explanation
    
    def plot_feature_importance_comparison(self, model_names, save_path=None):
        """
        Compare feature importance across multiple models.
        
        Args:
            model_names (list): List of model names to compare
            save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 4 * len(model_names)))
        
        if len(model_names) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            if model_name in self.shap_values:
                importance_df = self.get_feature_importance(model_name, top_n=10)
                
                axes[i].barh(range(len(importance_df)), importance_df['importance'])
                axes[i].set_yticks(range(len(importance_df)))
                axes[i].set_yticklabels(importance_df['feature'])
                axes[i].set_xlabel('Mean |SHAP value|')
                axes[i].set_title(f'Feature Importance - {model_name.upper()}')
                axes[i].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_explanation_report(self, X, model_name, sample_indices=None, save_path=None):
        """
        Create a comprehensive explanation report.
        
        Args:
            X (pd.DataFrame): Features to explain
            model_name (str): Name of the model
            sample_indices (list): List of sample indices to explain
            save_path (str): Path to save report
        """
        if sample_indices is None:
            sample_indices = [0, 1, 2]  # Default to first 3 samples
        
        report = []
        report.append("=" * 60)
        report.append(f"SHAP EXPLANATION REPORT - {model_name.upper()}")
        report.append("=" * 60)
        report.append("")
        
        # Feature importance
        importance_df = self.get_feature_importance(model_name, top_n=15)
        report.append("TOP 15 FEATURE IMPORTANCE:")
        report.append("-" * 30)
        for _, row in importance_df.iterrows():
            report.append(f"{row['feature']:25s}: {row['importance']:.4f}")
        report.append("")
        
        # Individual predictions
        for idx in sample_indices:
            explanation = self.explain_prediction(X, model_name, idx)
            
            report.append(f"SAMPLE {idx} EXPLANATION:")
            report.append("-" * 30)
            report.append(f"Base Value: {explanation['base_value']:.4f}")
            report.append(f"Prediction: {explanation['prediction']:.4f}")
            report.append("")
            report.append("Top 10 Feature Contributions:")
            
            for feature, contrib in explanation['sorted_features'][:10]:
                report.append(f"  {feature:20s}: {contrib['shap_value']:+.4f} (value: {contrib['feature_value']:.2f})")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text
    
    def save_explanations(self, filepath):
        """
        Save SHAP explanations to disk.
        
        Args:
            filepath (str): Path to save explanations
        """
        import joblib
        
        print(f"Saving SHAP explanations to {filepath}...")
        
        explanation_data = {
            'explainers': self.explainers,
            'shap_values': self.shap_values,
            'background_data': self.background_data,
            'config': self.config
        }
        
        joblib.dump(explanation_data, filepath)
        print("SHAP explanations saved successfully!")
    
    def load_explanations(self, filepath):
        """
        Load SHAP explanations from disk.
        
        Args:
            filepath (str): Path to load explanations from
        """
        import joblib
        
        print(f"Loading SHAP explanations from {filepath}...")
        
        explanation_data = joblib.load(filepath)
        
        self.explainers = explanation_data['explainers']
        self.shap_values = explanation_data['shap_values']
        self.background_data = explanation_data['background_data']
        
        print("SHAP explanations loaded successfully!")
        print(f"Available explainers: {list(self.explainers.keys())}") 