"""
Unified Interface: Main API combining all xAIUQ components.

Provides a single entry point for all explainability and uncertainty quantification.
"""

import torch
import numpy as np
from typing import Dict, Optional, Any
import warnings

from .variational_explainer import VariationalExplainer
from .feature_attribution import TimeSeriesFeatureAttribution
from .regime_explainer import RegimeExplainer
from .causal_analyzer import CausalAnalyzer
from .uncertainty_quantifier import UncertaintyQuantifier

warnings.filterwarnings('ignore')


class ExplanationReport:
    """Structured report containing all explanations and uncertainty metrics."""
    
    def __init__(self, series_id: str = None):
        """
        Initialize explanation report.
        
        Args:
            series_id: Optional series identifier
        """
        self.series_id = series_id
        self.regime_shifts = None
        self.feature_attribution = None
        self.regime_explanation = None
        self.external_signal = None
        self.uncertainty = None
        self.complete = {}
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'series_id': self.series_id,
            'regime_shifts': self.regime_shifts,
            'feature_attribution': self.feature_attribution,
            'regime_explanation': self.regime_explanation,
            'external_signal': self.external_signal,
            'uncertainty': self.uncertainty,
            'complete': self.complete
        }


class DelphiExplainer:
    """
    Main explainer class that orchestrates all xAIUQ components.
    
    Provides unified interface for:
    - Regime shift explanations
    - Feature attribution
    - Regime-specific explanations
    - External signal causal analysis
    - Uncertainty quantification
    """
    
    def __init__(self, model, device: str = 'cpu'):
        """
        Initialize Delphi explainer.
        
        Args:
            model: DELPHICore model instance
            device: Device for computation ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize component explainers
        self.variational_explainer = VariationalExplainer(n_states=model.n_states)
        self.feature_attribution = TimeSeriesFeatureAttribution()
        self.regime_explainer = RegimeExplainer()
        self.causal_analyzer = CausalAnalyzer()
        self.uncertainty_quantifier = UncertaintyQuantifier()
    
    def explain(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        include_uncertainty: bool = True,
        series_id: Optional[str] = None
    ) -> ExplanationReport:
        """
        Generate comprehensive explanation report.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            include_uncertainty: Whether to include uncertainty quantification
            series_id: Optional series identifier
        
        Returns:
            ExplanationReport with all explanations
        """
        # Ensure tensors are on correct device
        x = x.to(self.device)
        if parametric_forecast is not None:
            parametric_forecast = parametric_forecast.to(self.device)
        
        report = ExplanationReport(series_id=series_id)
        
        # Get regime shift explanations
        print("      Computing regime shifts...")
        report.regime_shifts = self.explain_regime_shifts(x, parametric_forecast)
        print("      ✓ Regime shifts complete")
        
        # Get feature attribution
        print("      Computing feature attribution (SHAP - this may take a while on CPU)...")
        report.feature_attribution = self.explain_features(x, parametric_forecast)
        print("      ✓ Feature attribution complete")
        
        # Get regime explanations
        print("      Computing regime explanations...")
        report.regime_explanation = self._get_regime_explanation(x, parametric_forecast)
        print("      ✓ Regime explanations complete")
        
        # Get external signal analysis
        print("      Computing external signal analysis...")
        report.external_signal = self.explain_external_signal(x, parametric_forecast)
        print("      ✓ External signal analysis complete")
        
        # Get uncertainty quantification
        if include_uncertainty:
            print("      Computing uncertainty quantification...")
            report.uncertainty = self.quantify_uncertainty(x, parametric_forecast)
            print("      ✓ Uncertainty quantification complete")
        
        # Combine all into complete report
        report.complete = report.to_dict()
        
        return report
    
    def explain_regime_shifts(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Explain regime shifts using variational HMM structure.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with regime shift explanations
        """
        x = x.to(self.device)
        if parametric_forecast is not None:
            parametric_forecast = parametric_forecast.to(self.device)
        
        return self.variational_explainer.explain_regime_shifts(
            self.model, x, parametric_forecast
        )
    
    def explain_features(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Explain feature contributions using SHAP-based attribution.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with feature attributions
        """
        x = x.to(self.device)
        if parametric_forecast is not None:
            parametric_forecast = parametric_forecast.to(self.device)
        
        return self.feature_attribution.attribute_features(
            self.model, x, parametric_forecast
        )
    
    def explain_external_signal(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Explain external signal (influencer data) causal impact.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with external signal analysis
        """
        x = x.to(self.device)
        if parametric_forecast is not None:
            parametric_forecast = parametric_forecast.to(self.device)
        
        causal_impact = self.causal_analyzer.causal_impact(
            self.model, x, parametric_forecast
        )
        
        signal_sensitivity = self.causal_analyzer.signal_sensitivity(
            self.model, x, parametric_forecast
        )
        
        return {
            'causal_impact': causal_impact,
            'signal_sensitivity': signal_sensitivity
        }
    
    def quantify_uncertainty(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Quantify uncertainty with aleatoric/epistemic decomposition.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with uncertainty metrics
        """
        x = x.to(self.device)
        if parametric_forecast is not None:
            parametric_forecast = parametric_forecast.to(self.device)
        
        # Get uncertainty decomposition
        uncertainty_decomp = self.uncertainty_quantifier.decompose_uncertainty(
            self.model, x, parametric_forecast
        )
        
        # Get regime uncertainty
        with torch.no_grad():
            results = self.model(x, parametric_forecast=parametric_forecast)
            state_probs = results['state_probs']
        
        regime_uncertainty = self.uncertainty_quantifier.regime_uncertainty(state_probs)
        
        # Get enhanced confidence intervals
        confidence_intervals = self.uncertainty_quantifier.enhanced_confidence_intervals(
            self.model, x, parametric_forecast
        )
        
        return {
            'uncertainty_decomposition': uncertainty_decomp,
            'regime_uncertainty': regime_uncertainty,
            'confidence_intervals': confidence_intervals
        }
    
    def _get_regime_explanation(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """Get regime-specific explanations."""
        x = x.to(self.device)
        if parametric_forecast is not None:
            parametric_forecast = parametric_forecast.to(self.device)
        
        with torch.no_grad():
            results = self.model(x, parametric_forecast=parametric_forecast, return_states=True)
            states = results['states']
            state_probs = results['state_probs']
        
        # State activation explanation
        activation_explanation = self.regime_explainer.explain_state_activation(
            self.model, x, states
        )
        
        # Corrector contributions
        corrector_contributions = self.regime_explainer.corrector_contributions(
            self.model, x, states, parametric_forecast
        )
        
        # Regime transitions
        transition_explanation = self.regime_explainer.regime_transition_explanation(
            state_probs, None
        )
        
        return {
            'activation': activation_explanation,
            'corrector_contributions': corrector_contributions,
            'transitions': transition_explanation
        }

