"""
Comprehensive evaluation module for fraud detection models.
Implements precision, recall, F1, ROC curves, and other metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class FraudEvaluator:
    """Comprehensive evaluator for fraud detection models."""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, probabilities, model_name):
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            probabilities (np.array): Prediction probabilities
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation results
        """
        print(f"Evaluating {model_name}...")
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate ROC metrics
        roc_auc = roc_auc_score(y_true, probabilities)
        fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
        
        # Calculate Precision-Recall metrics
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, probabilities)
        avg_precision = average_precision_score(y_true, probabilities)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'confusion_matrix': cm,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds
            },
            'precision_recall_curve': {
                'precision': precision_curve,
                'recall': recall_curve,
                'thresholds': pr_thresholds
            }
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        
        return results
    
    def evaluate_all_models(self, y_true, predictions_dict):
        """
        Evaluate multiple models.
        
        Args:
            y_true (np.array): True labels
            predictions_dict (dict): Dictionary of model predictions
            
        Returns:
            dict: Results for all models
        """
        print("Evaluating all models...")
        
        all_results = {}
        
        for model_name, pred_data in predictions_dict.items():
            y_pred = pred_data['predictions']
            probabilities = pred_data['probabilities']
            
            results = self.evaluate_model(y_true, y_pred, probabilities, model_name)
            all_results[model_name] = results
        
        return all_results
    
    def plot_roc_curves(self, save_path=None):
        """
        Plot ROC curves for all models.
        
        Args:
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            roc_data = results['roc_curve']
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'{model_name} (AUC = {results["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curves(self, save_path=None):
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            pr_data = results['precision_recall_curve']
            plt.plot(pr_data['recall'], pr_data['precision'], 
                    label=f'{model_name} (AP = {results["avg_precision"]:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrices(self, save_path=None):
        """
        Plot confusion matrices for all models.
        
        Args:
            save_path (str): Path to save plot
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name.upper()}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None):
        """
        Plot comparison of key metrics across models.
        
        Args:
            save_path (str): Path to save plot
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + 0.01,
                           f'{value:.3f}',
                           ha='center', va='bottom')
            
            axes[i].grid(axis='y', alpha=0.3)
        
        # Remove extra subplot
        axes[-1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_evaluation_report(self, save_path=None):
        """
        Create comprehensive evaluation report.
        
        Args:
            save_path (str): Path to save report
        """
        report = []
        report.append("=" * 60)
        report.append("FRAUD DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary table
        report.append("MODEL PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        report.append(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10}")
        report.append("-" * 70)
        
        for model_name, results in self.results.items():
            report.append(f"{model_name:<15} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                        f"{results['recall']:<10.4f} {results['f1_score']:<10.4f} {results['roc_auc']:<10.4f}")
        
        report.append("")
        
        # Detailed results for each model
        for model_name, results in self.results.items():
            report.append(f"DETAILED RESULTS - {model_name.upper()}:")
            report.append("-" * 40)
            report.append(f"Accuracy: {results['accuracy']:.4f}")
            report.append(f"Precision: {results['precision']:.4f}")
            report.append(f"Recall: {results['recall']:.4f}")
            report.append(f"F1-Score: {results['f1_score']:.4f}")
            report.append(f"ROC AUC: {results['roc_auc']:.4f}")
            report.append(f"Average Precision: {results['avg_precision']:.4f}")
            report.append(f"Specificity: {results['specificity']:.4f}")
            report.append(f"Sensitivity: {results['sensitivity']:.4f}")
            report.append("")
            
            # Confusion matrix
            cm = results['confusion_matrix']
            report.append("Confusion Matrix:")
            report.append(f"  True Negatives: {cm[0,0]}")
            report.append(f"  False Positives: {cm[0,1]}")
            report.append(f"  False Negatives: {cm[1,0]}")
            report.append(f"  True Positives: {cm[1,1]}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        best_f1_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_auc_model = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
        
        report.append(f"Best F1-Score: {best_f1_model[0]} ({best_f1_model[1]['f1_score']:.4f})")
        report.append(f"Best ROC AUC: {best_auc_model[0]} ({best_auc_model[1]['roc_auc']:.4f})")
        
        # Performance analysis
        report.append("")
        report.append("PERFORMANCE ANALYSIS:")
        report.append("-" * 20)
        
        for model_name, results in self.results.items():
            if results['precision'] < 0.8:
                report.append(f"- {model_name}: Low precision, consider adjusting threshold")
            if results['recall'] < 0.8:
                report.append(f"- {model_name}: Low recall, consider data balancing")
            if results['roc_auc'] > 0.9:
                report.append(f"- {model_name}: Excellent ROC AUC performance")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text
    
    def cross_validate_model(self, model, X, y, cv_folds=5):
        """
        Perform cross-validation for a model.
        
        Args:
            model: Trained model
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Cross-validation results
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validate with different metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        print("Cross-validation results:")
        for metric, results in cv_results.items():
            print(f"  {metric}: {results['mean']:.4f} (+/- {results['std'] * 2:.4f})")
        
        return cv_results
    
    def find_optimal_threshold(self, y_true, probabilities, metric='f1'):
        """
        Find optimal threshold for classification.
        
        Args:
            y_true (np.array): True labels
            probabilities (np.array): Prediction probabilities
            metric (str): Metric to optimize
            
        Returns:
            float: Optimal threshold
        """
        print(f"Finding optimal threshold using {metric}...")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        scores = []
        
        for threshold in thresholds:
            y_pred = (probabilities >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        optimal_threshold = thresholds[np.argmax(scores)]
        optimal_score = max(scores)
        
        print(f"Optimal threshold: {optimal_threshold:.3f} ({metric} = {optimal_score:.4f})")
        
        return optimal_threshold, optimal_score
    
    def plot_threshold_analysis(self, y_true, probabilities_dict, save_path=None):
        """
        Plot threshold analysis for all models.
        
        Args:
            y_true (np.array): True labels
            probabilities_dict (dict): Dictionary of model probabilities
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        
        for model_name, probabilities in probabilities_dict.items():
            f1_scores = []
            precision_scores = []
            recall_scores = []
            
            for threshold in thresholds:
                y_pred = (probabilities >= threshold).astype(int)
                f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
                precision_scores.append(precision_score(y_true, y_pred, zero_division=0))
                recall_scores.append(recall_score(y_true, y_pred, zero_division=0))
            
            plt.plot(thresholds, f1_scores, label=f'{model_name} (F1)')
        
        plt.xlabel('Threshold')
        plt.ylabel('F1-Score')
        plt.title('Threshold Analysis - F1-Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, filepath):
        """
        Save evaluation results to disk.
        
        Args:
            filepath (str): Path to save results
        """
        import joblib
        
        print(f"Saving evaluation results to {filepath}...")
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for model_name, results in self.results.items():
            results_to_save[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results_to_save[model_name][key] = value.tolist()
                else:
                    results_to_save[model_name][key] = value
        
        joblib.dump(results_to_save, filepath)
        print("Evaluation results saved successfully!")
    
    def load_results(self, filepath):
        """
        Load evaluation results from disk.
        
        Args:
            filepath (str): Path to load results from
        """
        import joblib
        
        print(f"Loading evaluation results from {filepath}...")
        
        results_data = joblib.load(filepath)
        
        # Convert lists back to numpy arrays
        for model_name, results in results_data.items():
            self.results[model_name] = {}
            for key, value in results.items():
                if key in ['roc_curve', 'precision_recall_curve']:
                    self.results[model_name][key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list):
                            self.results[model_name][key][subkey] = np.array(subvalue)
                        else:
                            self.results[model_name][key][subkey] = subvalue
                elif isinstance(value, list):
                    self.results[model_name][key] = np.array(value)
                else:
                    self.results[model_name][key] = value
        
        print("Evaluation results loaded successfully!")
        print(f"Available models: {list(self.results.keys())}") 