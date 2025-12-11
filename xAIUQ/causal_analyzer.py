"""
Causal Analyzer: Intervention-based analysis for external signal (weak signal ratio) impact.

Provides "what-if" scenarios and causal attribution for influencer data effects.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class CausalAnalyzer:
    """
    Analyzer for causal impact of external signals (weak signal ratio) on forecasts.
    
    Uses intervention analysis to:
    - Measure causal impact of influencer data
    - Test "what-if" scenarios
    - Quantify signal sensitivity
    """
    
    def __init__(self, n_interventions: int = 10):
        """
        Initialize causal analyzer.
        
        Args:
            n_interventions: Number of intervention points to test
        """
        self.n_interventions = n_interventions
    
    def intervention_analysis(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor],
        weak_signal_interventions: Optional[List[float]] = None
    ) -> Dict:
        """
        Perform intervention analysis: "what-if" scenarios for weak signal changes.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, 3) where x[:, :, 1] is weak signal ratio
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            weak_signal_interventions: List of intervention values to test (default: [0.0, 0.25, 0.5, 0.75, 1.0])
        
        Returns:
            Dictionary with intervention results
        """
        model.eval()
        
        if weak_signal_interventions is None:
            weak_signal_interventions = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        batch_size, seq_len, n_features = x.shape
        
        # Baseline forecast (original weak signal)
        with torch.no_grad():
            baseline_results = model(x, parametric_forecast=parametric_forecast)
            baseline_forecast = baseline_results['forecast']  # (batch, horizon)
        
        # Intervention results
        intervention_results = {}
        
        for intervention_value in weak_signal_interventions:
            # Create intervened input
            x_intervened = x.clone()
            x_intervened[:, :, 1] = intervention_value  # Set weak signal to intervention value
            
            # Get forecast with intervention
            with torch.no_grad():
                intervened_results = model(x_intervened, parametric_forecast=parametric_forecast)
                intervened_forecast = intervened_results['forecast']
            
            # Compute difference
            forecast_diff = intervened_forecast - baseline_forecast
            
            intervention_results[f'weak_signal_{intervention_value:.2f}'] = {
                'forecast': intervened_forecast.cpu().numpy(),
                'forecast_diff': forecast_diff.cpu().numpy(),
                'mean_diff': float(forecast_diff.mean().item()),
                'max_diff': float(forecast_diff.abs().max().item()),
                'intervention_value': intervention_value
            }
        
        return {
            'baseline_forecast': baseline_forecast.cpu().numpy(),
            'interventions': intervention_results,
            'intervention_values': weak_signal_interventions
        }
    
    def causal_impact(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Measure causal impact of influencer data on forecasts.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, 3)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with causal impact metrics
        """
        model.eval()
        
        batch_size, seq_len, n_features = x.shape
        
        # Get original forecast
        with torch.no_grad():
            original_results = model(x, parametric_forecast=parametric_forecast)
            original_forecast = original_results['forecast']
            original_correction = original_results['correction']
        
        # Counterfactual: Remove weak signal (set to neutral 0.5)
        x_no_weak = x.clone()
        x_no_weak[:, :, 1] = 0.5  # Neutral weak signal
        
        with torch.no_grad():
            no_weak_results = model(x_no_weak, parametric_forecast=parametric_forecast)
            no_weak_forecast = no_weak_results['forecast']
            no_weak_correction = no_weak_results['correction']
        
        # Compute causal impact
        causal_effect = original_forecast - no_weak_forecast  # (batch, horizon)
        
        # Get state probabilities to see if External Signal state is active
        state_probs = original_results['state_probs']  # (batch, horizon, n_states)
        external_signal_prob = state_probs[:, :, 3]  # State 3 is External Signal
        
        # Compute impact metrics
        mean_causal_effect = causal_effect.mean().item()
        max_causal_effect = causal_effect.abs().max().item()
        
        # Correlation between weak signal and causal effect
        weak_signal_values = x[:, -1, 1].unsqueeze(1).expand(-1, causal_effect.shape[1])  # (batch, horizon)
        weak_signal_flat = weak_signal_values.flatten()
        causal_effect_flat = causal_effect.flatten()
        
        # Compute correlation with proper edge case handling
        if causal_effect_flat.numel() < 2:
            correlation = 0.0
        else:
            # Stack as (2, N) for torch.corrcoef: rows are variables, columns are observations
            stacked = torch.stack([weak_signal_flat, causal_effect_flat], dim=0)  # (2, N)
            
            # Check for constant values (zero variance)
            if stacked.std(dim=1).min() < 1e-8:
                correlation = 0.0
            else:
                corr_matrix = torch.corrcoef(stacked)  # (2, 2)
                correlation = corr_matrix[0, 1].item()
                
                # Handle NaN (can occur with numerical issues)
                if torch.isnan(torch.tensor(correlation)):
                    correlation = 0.0
        
        return {
            'causal_effect': causal_effect.cpu().numpy(),
            'mean_causal_effect': mean_causal_effect,
            'max_causal_effect': max_causal_effect,
            'correlation_with_weak_signal': correlation,
            'external_signal_probability': external_signal_prob.cpu().numpy(),
            'original_forecast': original_forecast.cpu().numpy(),
            'no_weak_forecast': no_weak_forecast.cpu().numpy(),
            'original_correction': original_correction.cpu().numpy(),
            'no_weak_correction': no_weak_correction.cpu().numpy()
        }
    
    def signal_sensitivity(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        signal_range: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        Quantify how forecast changes with weak signal variations.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, 3)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            signal_range: Range of weak signal values to test (default: [0.0, 1.0])
        
        Returns:
            Dictionary with sensitivity analysis
        """
        model.eval()
        
        if signal_range is None:
            signal_range = (0.0, 1.0)
        
        # Test multiple signal values
        signal_values = np.linspace(signal_range[0], signal_range[1], self.n_interventions)
        
        batch_size, seq_len, n_features = x.shape
        
        # Get baseline (original signal value)
        original_signal = x[:, -1, 1].mean().item()  # Average weak signal value
        
        sensitivity_results = []
        
        for signal_val in signal_values:
            x_test = x.clone()
            x_test[:, :, 1] = signal_val  # Set weak signal
            
            with torch.no_grad():
                results = model(x_test, parametric_forecast=parametric_forecast)
                forecast = results['forecast']
            
            sensitivity_results.append({
                'signal_value': float(signal_val),
                'forecast_mean': float(forecast.mean().item()),
                'forecast_std': float(forecast.std().item()),
                'forecast': forecast.cpu().numpy()
            })
        
        # Compute sensitivity (derivative approximation)
        signal_values_array = np.array([r['signal_value'] for r in sensitivity_results])
        forecast_means = np.array([r['forecast_mean'] for r in sensitivity_results])
        
        # Approximate derivative
        if len(signal_values_array) > 1:
            sensitivity_gradient = np.gradient(forecast_means, signal_values_array)
        else:
            sensitivity_gradient = np.array([0.0])
        
        # Find maximum sensitivity point
        max_sensitivity_idx = np.argmax(np.abs(sensitivity_gradient))
        max_sensitivity_value = signal_values_array[max_sensitivity_idx]
        
        return {
            'sensitivity_results': sensitivity_results,
            'sensitivity_gradient': sensitivity_gradient.tolist(),
            'max_sensitivity_point': float(max_sensitivity_value),
            'max_sensitivity_value': float(np.abs(sensitivity_gradient).max()),
            'original_signal_value': original_signal,
            'signal_range': signal_range
        }

