"""
Comprehensive evaluation script for multimodal emotion analysis.
Includes metrics calculation, visualization, and model comparison.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_curve, roc_curve, auc
)
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils.config import EVALUATION_CONFIG, FACIAL_CONFIG, TEXT_CONFIG

class EmotionEvaluator:
    """Comprehensive evaluator for emotion analysis models."""
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_proba, emotions, model_name):
        """Calculate comprehensive metrics for a model."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=emotions, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        
        # ROC curves for each class
        roc_data = {}
        for i, emotion in enumerate(emotions):
            if len(np.unique(y_true)) > 1:  # Only if we have multiple classes
                fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                roc_data[emotion] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc
                }
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_data': roc_data,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return self.results[model_name]
    
    def plot_confusion_matrix(self, cm, emotions, model_name, save_path=None):
        """Plot confusion matrix with enhanced styling."""
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=emotions,
            yticklabels=emotions,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('Actual Emotion', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, roc_data, model_name, save_path=None):
        """Plot ROC curves for all emotions."""
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(roc_data)))
        
        for i, (emotion, data) in enumerate(roc_data.items()):
            plt.plot(
                data['fpr'], 
                data['tpr'], 
                color=colors[i],
                lw=2,
                label=f'{emotion} (AUC = {data["auc"]:.3f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {model_name.upper()}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, models_results, save_path=None):
        """Plot comparison of different metrics across models."""
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(models_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [models_results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01,
                    f'{value:.3f}', 
                    ha='center', 
                    va='bottom'
                )
            
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, models_results, emotions):
        """Create interactive dashboard using Plotly."""
        
        # Prepare data for plotting
        model_names = list(models_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, metric in enumerate(metrics):
            values = [models_results[model][metric] for model in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    marker_color=colors,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                ),
                row=(i // 2) + 1, col=(i % 2) + 1
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            showlegend=False,
            height=800
        )
        
        # Save as HTML
        fig.write_html("results/interactive_dashboard.html")
        
        return fig
    
    def generate_detailed_report(self, models_results, save_path=None):
        """Generate detailed evaluation report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MULTIMODAL EMOTION ANALYSIS - DETAILED EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary table
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        report_lines.append("-" * 50)
        
        for model_name, results in models_results.items():
            report_lines.append(
                f"{model_name:<15} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                f"{results['recall']:<10.4f} {results['f1_score']:<10.4f}"
            )
        
        report_lines.append("")
        
        # Detailed per-model analysis
        for model_name, results in models_results.items():
            report_lines.append(f"DETAILED ANALYSIS - {model_name.upper()}")
            report_lines.append("=" * 50)
            report_lines.append(f"Overall Accuracy: {results['accuracy']:.4f}")
            report_lines.append(f"Weighted Precision: {results['precision']:.4f}")
            report_lines.append(f"Weighted Recall: {results['recall']:.4f}")
            report_lines.append(f"Weighted F1-Score: {results['f1_score']:.4f}")
            report_lines.append("")
            
            # Per-class metrics
            report_lines.append("Per-Class Metrics:")
            report_lines.append("-" * 30)
            
            for emotion, metrics in results['classification_report'].items():
                if isinstance(metrics, dict) and emotion not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_lines.append(f"{emotion}:")
                    report_lines.append(f"  Precision: {metrics['precision']:.4f}")
                    report_lines.append(f"  Recall: {metrics['recall']:.4f}")
                    report_lines.append(f"  F1-Score: {metrics['f1-score']:.4f}")
                    report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("=" * 30)
        
        best_model = max(models_results.items(), key=lambda x: x[1]['f1_score'])
        report_lines.append(f"Best performing model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
        
        if 'multimodal' in models_results:
            multimodal_f1 = models_results['multimodal']['f1_score']
            individual_f1s = [results['f1_score'] for name, results in models_results.items() if name != 'multimodal']
            avg_individual_f1 = np.mean(individual_f1s)
            
            if multimodal_f1 > avg_individual_f1:
                improvement = ((multimodal_f1 - avg_individual_f1) / avg_individual_f1) * 100
                report_lines.append(f"Multimodal fusion provides {improvement:.2f}% improvement over individual models")
            else:
                report_lines.append("Multimodal fusion does not show significant improvement")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text
    
    def save_results(self, save_path="results/evaluation_results.json"):
        """Save all evaluation results to JSON file."""
        
        # Convert numpy arrays to lists for JSON serialization
        results_for_json = {}
        for model_name, results in self.results.items():
            results_for_json[model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'classification_report': results['classification_report'],
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'roc_data': {
                    emotion: {
                        'fpr': data['fpr'].tolist(),
                        'tpr': data['tpr'].tolist(),
                        'auc': float(data['auc'])
                    }
                    for emotion, data in results['roc_data'].items()
                }
            }
        
        with open(save_path, 'w') as f:
            json.dump(results_for_json, f, indent=4)
        
        print(f"Results saved to {save_path}")

def main():
    """Main evaluation function."""
    
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = EmotionEvaluator()
    
    # Load results from training (if available)
    results_files = [
        "results/facial_evaluation_results.json",
        "results/text_rnn_evaluation_results.json",
        "results/multimodal_evaluation_results.json"
    ]
    
    models_results = {}
    
    for file_path in results_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                model_name = file_path.split('/')[-1].split('_')[0]
                models_results[model_name] = data
    
    if not models_results:
        print("No evaluation results found. Please run training scripts first.")
        return
    
    # Generate visualizations and reports
    print("Generating visualizations and reports...")
    
    # Plot confusion matrices
    for model_name, results in models_results.items():
        if 'confusion_matrix' in results:
            emotions = FACIAL_CONFIG["emotions"] if model_name == "facial" else TEXT_CONFIG["emotions"]
            evaluator.plot_confusion_matrix(
                np.array(results['confusion_matrix']),
                emotions,
                model_name,
                save_path=f"results/{model_name}_confusion_matrix.png"
            )
    
    # Plot ROC curves
    for model_name, results in models_results.items():
        if 'roc_data' in results:
            evaluator.plot_roc_curves(
                results['roc_data'],
                model_name,
                save_path=f"results/{model_name}_roc_curves.png"
            )
    
    # Plot metrics comparison
    evaluator.plot_metrics_comparison(
        models_results,
        save_path="results/metrics_comparison.png"
    )
    
    # Create interactive dashboard
    emotions = FACIAL_CONFIG["emotions"]  # Use facial emotions as default
    evaluator.create_interactive_dashboard(models_results, emotions)
    
    # Generate detailed report
    evaluator.generate_detailed_report(
        models_results,
        save_path="results/detailed_evaluation_report.txt"
    )
    
    # Save all results
    evaluator.results = models_results
    evaluator.save_results()
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 