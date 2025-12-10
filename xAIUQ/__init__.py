"""
xAIUQ: Hybrid Explainability and Uncertainty Quantification Layer for DELPHI.

This module provides comprehensive explainability and uncertainty quantification
for DELPHI model predictions, leveraging the existing variational inference framework.
"""

from .variational_explainer import VariationalExplainer
from .feature_attribution import TimeSeriesFeatureAttribution
from .regime_explainer import RegimeExplainer
from .causal_analyzer import CausalAnalyzer
from .uncertainty_quantifier import UncertaintyQuantifier
from .unified_interface import DelphiExplainer, ExplanationReport
from . import result_loader

__all__ = [
    'VariationalExplainer',
    'TimeSeriesFeatureAttribution',
    'RegimeExplainer',
    'CausalAnalyzer',
    'UncertaintyQuantifier',
    'DelphiExplainer',
    'ExplanationReport',
    'result_loader'
]

