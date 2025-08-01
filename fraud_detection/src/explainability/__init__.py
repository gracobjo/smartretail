"""
Explainability module for fraud detection models.

This module provides SHAP-based explanations and interpretability tools
for fraud detection models.
"""

from .shap_explainer import SHAPExplainer

__all__ = ['SHAPExplainer'] 