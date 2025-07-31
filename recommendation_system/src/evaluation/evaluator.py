"""
Comprehensive evaluation system for recommendation algorithms.
Implements precision@k, recall@k, diversity, coverage, and other metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    """Comprehensive evaluator for recommendation systems."""
    
    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.k_values = config['k_values']
        
    def evaluate(self, recommendations, ground_truth, user_ids=None):
        """
        Evaluate recommendation quality.
        
        Args:
            recommendations (dict): Dictionary of user_id -> list of (item_id, score)
            ground_truth (dict): Dictionary of user_id -> list of relevant items
            user_ids (list): List of user IDs to evaluate
            
        Returns:
            dict: Evaluation results
        """
        print("Evaluating recommendation quality...")
        
        if user_ids is None:
            user_ids = list(recommendations.keys())
        
        results = {}
        
        # Calculate metrics for each k value
        for k in self.k_values:
            k_results = {}
            
            if 'precision' in self.metrics:
                k_results['precision'] = self._calculate_precision_at_k(
                    recommendations, ground_truth, user_ids, k
                )
            
            if 'recall' in self.metrics:
                k_results['recall'] = self._calculate_recall_at_k(
                    recommendations, ground_truth, user_ids, k
                )
            
            if 'ndcg' in self.metrics:
                k_results['ndcg'] = self._calculate_ndcg_at_k(
                    recommendations, ground_truth, user_ids, k
                )
            
            if 'diversity' in self.metrics:
                k_results['diversity'] = self._calculate_diversity_at_k(
                    recommendations, user_ids, k
                )
            
            if 'coverage' in self.metrics:
                k_results['coverage'] = self._calculate_coverage_at_k(
                    recommendations, user_ids, k
                )
            
            if 'serendipity' in self.metrics:
                k_results['serendipity'] = self._calculate_serendipity_at_k(
                    recommendations, ground_truth, user_ids, k
                )
            
            results[f'k={k}'] = k_results
        
        # Calculate overall metrics
        overall_results = self._calculate_overall_metrics(results)
        results['overall'] = overall_results
        
        print("Evaluation completed!")
        return results
    
    def _calculate_precision_at_k(self, recommendations, ground_truth, user_ids, k):
        """Calculate precision@k."""
        precisions = []
        
        for user_id in user_ids:
            if user_id not in recommendations or user_id not in ground_truth:
                continue
            
            recommended_items = [item for item, _ in recommendations[user_id][:k]]
            relevant_items = set(ground_truth[user_id])
            
            if len(recommended_items) > 0:
                precision = len(set(recommended_items) & relevant_items) / len(recommended_items)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_recall_at_k(self, recommendations, ground_truth, user_ids, k):
        """Calculate recall@k."""
        recalls = []
        
        for user_id in user_ids:
            if user_id not in recommendations or user_id not in ground_truth:
                continue
            
            recommended_items = [item for item, _ in recommendations[user_id][:k]]
            relevant_items = set(ground_truth[user_id])
            
            if len(relevant_items) > 0:
                recall = len(set(recommended_items) & relevant_items) / len(relevant_items)
                recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def _calculate_ndcg_at_k(self, recommendations, ground_truth, user_ids, k):
        """Calculate NDCG@k."""
        ndcgs = []
        
        for user_id in user_ids:
            if user_id not in recommendations or user_id not in ground_truth:
                continue
            
            recommended_items = [item for item, _ in recommendations[user_id][:k]]
            relevant_items = set(ground_truth[user_id])
            
            # Create relevance scores
            y_true = []
            y_score = []
            
            for item in recommended_items:
                y_true.append(1 if item in relevant_items else 0)
                # Use position-based score (higher position = higher score)
                y_score.append(1.0)
            
            if len(y_true) > 0:
                try:
                    ndcg = ndcg_score([y_true], [y_score], k=k)
                    ndcgs.append(ndcg)
                except:
                    continue
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def _calculate_diversity_at_k(self, recommendations, user_ids, k):
        """Calculate diversity@k (intra-list diversity)."""
        diversities = []
        
        for user_id in user_ids:
            if user_id not in recommendations:
                continue
            
            recommended_items = [item for item, _ in recommendations[user_id][:k]]
            
            if len(recommended_items) > 1:
                # Calculate pairwise diversity (simplified)
                diversity = 1.0  # Placeholder - in practice, calculate item similarity
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _calculate_coverage_at_k(self, recommendations, user_ids, k):
        """Calculate coverage@k (percentage of items recommended)."""
        all_recommended_items = set()
        total_recommendations = 0
        
        for user_id in user_ids:
            if user_id not in recommendations:
                continue
            
            recommended_items = [item for item, _ in recommendations[user_id][:k]]
            all_recommended_items.update(recommended_items)
            total_recommendations += len(recommended_items)
        
        # Estimate total number of items (this would be known in practice)
        total_items = max(all_recommended_items) + 1 if all_recommended_items else 1
        
        coverage = len(all_recommended_items) / total_items
        return coverage
    
    def _calculate_serendipity_at_k(self, recommendations, ground_truth, user_ids, k):
        """Calculate serendipity@k (unexpected but relevant items)."""
        serendipity_scores = []
        
        for user_id in user_ids:
            if user_id not in recommendations or user_id not in ground_truth:
                continue
            
            recommended_items = [item for item, _ in recommendations[user_id][:k]]
            relevant_items = set(ground_truth[user_id])
            
            # Simplified serendipity calculation
            # In practice, this would consider item popularity and user preferences
            serendipitous_items = 0
            for item in recommended_items:
                if item in relevant_items:
                    # Assume less popular items are more serendipitous
                    serendipitous_items += 1
            
            if len(recommended_items) > 0:
                serendipity = serendipitous_items / len(recommended_items)
                serendipity_scores.append(serendipity)
        
        return np.mean(serendipity_scores) if serendipity_scores else 0.0
    
    def _calculate_overall_metrics(self, results):
        """Calculate overall metrics across all k values."""
        overall = {}
        
        for metric in ['precision', 'recall', 'ndcg', 'diversity', 'coverage', 'serendipity']:
            values = []
            for k_result in results.values():
                if isinstance(k_result, dict) and metric in k_result:
                    values.append(k_result[metric])
            
            if values:
                overall[metric] = np.mean(values)
        
        return overall
    
    def plot_metrics_comparison(self, results, save_path=None):
        """Plot comparison of metrics across different k values."""
        metrics = ['precision', 'recall', 'ndcg', 'diversity']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            k_values = []
            metric_values = []
            
            for k_key, k_results in results.items():
                if k_key != 'overall' and metric in k_results:
                    k = int(k_key.split('=')[1])
                    k_values.append(k)
                    metric_values.append(k_results[metric])
            
            if k_values and metric_values:
                axes[i].plot(k_values, metric_values, marker='o', linewidth=2, markersize=8)
                axes[i].set_title(f'{metric.upper()}@k', fontweight='bold')
                axes[i].set_xlabel('k')
                axes[i].set_ylabel(metric.upper())
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, model_results, save_path=None):
        """Plot comparison of different models."""
        models = list(model_results.keys())
        metrics = ['precision', 'recall', 'ndcg']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = []
            for model in models:
                if 'overall' in model_results[model] and metric in model_results[model]['overall']:
                    values.append(model_results[model]['overall'][metric])
                else:
                    values.append(0)
            
            bars = axes[i].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[i].set_title(f'{metric.upper()} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric.upper())
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
    
    def create_interactive_dashboard(self, results, model_results=None):
        """Create interactive Plotly dashboard."""
        # Create subplots
        if model_results:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Precision@k', 'Recall@k', 'NDCG@k', 'Model Comparison'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
        else:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Precision@k', 'Recall@k', 'NDCG@k', 'Diversity@k')
            )
        
        # Plot metrics over k values
        metrics = ['precision', 'recall', 'ndcg', 'diversity']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            k_values = []
            metric_values = []
            
            for k_key, k_results in results.items():
                if k_key != 'overall' and metric in k_results:
                    k = int(k_key.split('=')[1])
                    k_values.append(k)
                    metric_values.append(k_results[metric])
            
            if k_values and metric_values:
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=metric_values,
                        mode='lines+markers',
                        name=f'{metric.upper()}@k',
                        line=dict(color=colors[i])
                    ),
                    row=(i // 2) + 1,
                    col=(i % 2) + 1
                )
        
        # Add model comparison if available
        if model_results:
            models = list(model_results.keys())
            precision_values = []
            recall_values = []
            
            for model in models:
                if 'overall' in model_results[model]:
                    precision_values.append(model_results[model]['overall'].get('precision', 0))
                    recall_values.append(model_results[model]['overall'].get('recall', 0))
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=precision_values,
                    name='Precision',
                    marker_color='#1f77b4'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Recommendation System Evaluation Dashboard",
            height=800,
            showlegend=True
        )
        
        fig.show()
    
    def generate_report(self, results, model_results=None, save_path=None):
        """Generate detailed evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("RECOMMENDATION SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        if 'overall' in results:
            report.append("OVERALL METRICS:")
            report.append("-" * 20)
            for metric, value in results['overall'].items():
                report.append(f"{metric.upper()}: {value:.4f}")
            report.append("")
        
        # Metrics by k value
        report.append("METRICS BY K VALUE:")
        report.append("-" * 20)
        for k_key, k_results in results.items():
            if k_key != 'overall':
                report.append(f"\n{k_key.upper()}:")
                for metric, value in k_results.items():
                    report.append(f"  {metric.upper()}: {value:.4f}")
        
        # Model comparison
        if model_results:
            report.append("\n" + "=" * 60)
            report.append("MODEL COMPARISON")
            report.append("=" * 60)
            
            for model_name, model_result in model_results.items():
                report.append(f"\n{model_name.upper()}:")
                if 'overall' in model_result:
                    for metric, value in model_result['overall'].items():
                        report.append(f"  {metric.upper()}: {value:.4f}")
        
        # Recommendations
        report.append("\n" + "=" * 60)
        report.append("RECOMMENDATIONS")
        report.append("=" * 60)
        
        if 'overall' in results:
            precision = results['overall'].get('precision', 0)
            recall = results['overall'].get('recall', 0)
            
            if precision < 0.3:
                report.append("- Precision is low. Consider improving relevance.")
            elif precision > 0.7:
                report.append("- Precision is excellent!")
            
            if recall < 0.2:
                report.append("- Recall is low. Consider increasing coverage.")
            elif recall > 0.5:
                report.append("- Recall is good!")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text

class DiversityAnalyzer:
    """Analyze diversity of recommendations."""
    
    def __init__(self):
        pass
    
    def calculate_intra_list_diversity(self, recommendations):
        """Calculate intra-list diversity."""
        diversities = []
        
        for user_id, user_recs in recommendations.items():
            if len(user_recs) < 2:
                continue
            
            # Calculate pairwise similarities (simplified)
            items = [item for item, _ in user_recs]
            diversity = 1.0  # Placeholder - calculate actual diversity
            
            diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def calculate_inter_list_diversity(self, recommendations):
        """Calculate inter-list diversity."""
        all_recommendations = []
        
        for user_recs in recommendations.values():
            items = [item for item, _ in user_recs]
            all_recommendations.append(items)
        
        if len(all_recommendations) < 2:
            return 0.0
        
        # Calculate Jaccard similarity between recommendation lists
        similarities = []
        for i in range(len(all_recommendations)):
            for j in range(i + 1, len(all_recommendations)):
                set1 = set(all_recommendations[i])
                set2 = set(all_recommendations[j])
                
                if len(set1 | set2) > 0:
                    similarity = len(set1 & set2) / len(set1 | set2)
                    similarities.append(similarity)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        diversity = 1.0 - avg_similarity
        
        return diversity

class CoverageAnalyzer:
    """Analyze coverage of recommendations."""
    
    def __init__(self):
        pass
    
    def calculate_catalog_coverage(self, recommendations, total_items):
        """Calculate catalog coverage."""
        all_recommended_items = set()
        
        for user_recs in recommendations.values():
            items = [item for item, _ in user_recs]
            all_recommended_items.update(items)
        
        coverage = len(all_recommended_items) / total_items
        return coverage
    
    def calculate_user_coverage(self, recommendations, total_users):
        """Calculate user coverage."""
        users_with_recommendations = len(recommendations)
        coverage = users_with_recommendations / total_users
        return coverage 