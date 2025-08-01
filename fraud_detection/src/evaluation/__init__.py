"""
Evaluation module for fraud detection models.

This module provides comprehensive evaluation capabilities for fraud detection models,
including precision, recall, F1-score, ROC curves, and other metrics.
"""

from .evaluator import FraudEvaluator

__all__ = ['FraudEvaluator'] 