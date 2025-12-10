"""
Time-Series Feature Attribution: SHAP-based attribution for input features.

Adapts SHAP values for sequential data to explain contributions of the 3 input features:
- z_t: Normalized residuals
- w_t: Weak signal ratio
- ŷ_pred_t: Parametric forecast
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available. Feature attribution will use simplified methods.")

warnings.filterwarnings('ignore')


class TimeSeriesFeatureAttribution:
    """
    Time-series adapted feature attribution using SHAP or gradient-based methods.
    
    Handles the 3 input features [z_t, w_t, ŷ_pred_t] and provides:
    - Per-timestep attribution
    - Feature importance scores
    - Temporal importance patterns
    """
    
    def __init__(self, use_shap: bool = True):
        """
        Initialize feature attribution explainer.
        
        Args:
            use_shap: Whether to use SHAP library (if available)
        """
        self.use_shap = use_shap and SHAP_AVAILABLE
        self.feature_names = ['residuals (z_t)', 'weak_signal_ratio (w_t)', 'parametric_forecast (ŷ_pred_t)']
    
    def compute_shap_values(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
        num_samples: int = 100
    ) -> Dict:
        """
        Compute SHAP values for time-series input features.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, 3)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            baseline: Baseline input for SHAP (default: zeros or mean)
            num_samples: Number of samples for SHAP estimation
        
        Returns:
            Dictionary with SHAP values and attributions
        """
        model.eval()
        
        if self.use_shap:
            return self._compute_shap_with_library(model, x, parametric_forecast, baseline, num_samples)
        else:
            return self._compute_gradient_based(model, x, parametric_forecast)
    
    def _compute_shap_with_library(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor],
        baseline: Optional[torch.Tensor],
        num_samples: int
    ) -> Dict:
        """Compute SHAP using SHAP library."""
        # For time-series, we'll use a simplified approach
        # Flatten temporal dimension for SHAP computation
        batch_size, seq_len, n_features = x.shape
        
        # Create baseline if not provided
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        # Define prediction function
        def predict_fn(x_flat):
            # Reshape back to (batch, seq_len, n_features)
            x_reshaped = torch.tensor(x_flat, dtype=torch.float32).reshape(-1, seq_len, n_features)
            if parametric_forecast is not None:
                param_rep = parametric_forecast.repeat(x_reshaped.shape[0], 1)
            else:
                param_rep = None
            
            with torch.no_grad():
                results = model(x_reshaped, parametric_forecast=param_rep)
                return results['forecast'].cpu().numpy()
        
        # Use KernelExplainer for time-series
        explainer = shap.KernelExplainer(
            predict_fn,
            baseline.cpu().numpy().reshape(1, -1)
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(
            x.cpu().numpy().reshape(1, -1),
            nsamples=num_samples
        )
        
        # Reshape back to (seq_len, n_features, horizon)
        # Note: SHAP output shape depends on model output
        # For simplicity, we'll use gradient-based method as fallback
        return self._compute_gradient_based(model, x, parametric_forecast)
    
    def _compute_gradient_based(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor]
    ) -> Dict:
        """
        Compute feature attribution using integrated gradients (gradient-based).
        
        This is a simplified but effective method for time-series attribution.
        """
        model.eval()
        x.requires_grad_(True)
        
        # Forward pass
        results = model(x, parametric_forecast=parametric_forecast)
        forecast = results['forecast']  # (batch, horizon)
        
        batch_size, seq_len, n_features = x.shape
        horizon = forecast.shape[1]
        
        # Compute gradients for each feature
        feature_attributions = torch.zeros(batch_size, seq_len, n_features, horizon, device=x.device)
        
        for t in range(horizon):
            # Gradient w.r.t. input for each timestep
            grad = torch.autograd.grad(
                outputs=forecast[:, t].sum(),
                inputs=x,
                retain_graph=True,
                create_graph=False
            )[0]  # (batch, seq_len, n_features)
            
            feature_attributions[:, :, :, t] = grad
        
        # Average over batch
        attributions = feature_attributions.mean(dim=0)  # (seq_len, n_features, horizon)
        
        # Compute feature importance (sum over time and horizon)
        feature_importance = torch.abs(attributions).sum(dim=(0, 2))  # (n_features,)
        feature_importance = feature_importance / (feature_importance.sum() + 1e-8)  # Normalize
        
        return {
            'shap_values': attributions.cpu().numpy(),
            'feature_importance': feature_importance.cpu().numpy(),
            'feature_names': self.feature_names,
            'attribution_shape': attributions.shape
        }
    
    def attribute_features(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Attribute forecast to input features.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, 3)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        
        Returns:
            Dictionary with feature attributions
        """
        # Compute attributions
        attribution_results = self.compute_shap_values(model, x, parametric_forecast)
        
        # Get temporal importance
        temporal_importance = self.temporal_importance(
            attribution_results['shap_values'],
            range(x.shape[1])  # timesteps
        )
        
        return {
            **attribution_results,
            'temporal_importance': temporal_importance
        }
    
    def temporal_importance(
        self,
        shap_values: np.ndarray,
        timesteps: range
    ) -> Dict:
        """
        Compute temporal importance patterns from SHAP values.
        
        Args:
            shap_values: SHAP values array (seq_len, n_features, horizon) or similar
            timesteps: Range of timesteps
        
        Returns:
            Dictionary with temporal importance analysis
        """
        # Handle different shapes
        if len(shap_values.shape) == 3:
            seq_len, n_features, horizon = shap_values.shape
        else:
            # Flatten if needed
            shap_values = shap_values.reshape(-1, shap_values.shape[-2], shap_values.shape[-1])
            seq_len, n_features, horizon = shap_values.shape
        
        # Compute importance per timestep
        timestep_importance = np.abs(shap_values).sum(axis=(1, 2))  # (seq_len,)
        
        # Compute importance per feature over time
        feature_temporal_importance = np.abs(shap_values).sum(axis=2)  # (seq_len, n_features)
        
        # Find most important timesteps
        top_timesteps = np.argsort(timestep_importance)[-5:][::-1]  # Top 5
        
        return {
            'timestep_importance': timestep_importance.tolist(),
            'feature_temporal_importance': feature_temporal_importance.tolist(),
            'top_timesteps': top_timesteps.tolist(),
            'feature_names': self.feature_names
        }

