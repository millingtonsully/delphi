"""
Time-Series Feature Attribution: SHAP-based attribution for input features.

Uses GradientSHAP (Expected Gradients) to compute true Shapley values for the 3 input features:
- z_t: Normalized residuals
- w_t: Weak signal ratio
- ŷ_pred_t: Parametric forecast
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import warnings
import gc

try:
    import shap
except ImportError:
    raise ImportError(
        "SHAP library is required for feature attribution. "
        "Install with: pip install shap>=0.44.0"
    )

warnings.filterwarnings('ignore')


class DELPHIModelWrapper(nn.Module):
    """
    Wrapper around DELPHI model for SHAP GradientExplainer.
    
    GradientExplainer requires a nn.Module that takes input tensor
    and returns output tensor directly.
    
    Includes training mode lock to prevent SHAP from switching to eval mode,
    which is required for cuDNN LSTM gradient computation.
    """
    
    def __init__(self, model, parametric_forecast: Optional[torch.Tensor] = None):
        """
        Initialize wrapper.
        
        Args:
            model: DELPHICore model instance
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
        """
        super().__init__()
        self.model = model
        self.parametric_forecast = parametric_forecast
        self._training_locked = False  # Lock flag for training mode
    
    def lock_training_mode(self):
        """
        Lock wrapper and all children in training mode for gradient computation.
        
        This prevents SHAP's internal eval() calls from switching the model
        out of training mode, which is required for cuDNN LSTM backward pass.
        """
        self._training_locked = True
        # Force training mode on entire module tree
        for module in self.modules():
            module.training = True
    
    def unlock_training_mode(self):
        """Unlock training mode, allowing normal train/eval switching."""
        self._training_locked = False
    
    def train(self, mode: bool = True):
        """Override train to respect training lock."""
        if self._training_locked:
            return self  # Ignore mode changes when locked
        return super().train(mode)
    
    def eval(self):
        """Override eval to respect training lock."""
        if self._training_locked:
            return self  # Ignore eval calls when locked
        return super().eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning forecast tensor.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
        
        Returns:
            Forecast tensor (batch, horizon)
        """
        # Handle parametric forecast expansion for different batch sizes
        if self.parametric_forecast is not None:
            batch_size = x.shape[0]
            param_batch_size = self.parametric_forecast.shape[0]
            
            if batch_size != param_batch_size:
                # Expand parametric forecast to match batch size
                param_forecast = self.parametric_forecast[0:1].expand(batch_size, -1)
            else:
                param_forecast = self.parametric_forecast
        else:
            param_forecast = None
        
        results = self.model(x, parametric_forecast=param_forecast)
        return results['forecast']


class TimeSeriesFeatureAttribution:
    """
    Time-series feature attribution using GradientSHAP (Expected Gradients).
    
    Computes true Shapley values for the 3 input features [z_t, w_t, ŷ_pred_t]:
    - Handles temporal structure of (batch, seq_len, 3) inputs
    - Provides per-timestep attribution
    - Computes feature importance scores
    - Analyzes temporal importance patterns
    """
    
    def __init__(self):
        """Initialize feature attribution explainer."""
        self.feature_names = [
            'residuals (z_t)', 
            'weak_signal_ratio (w_t)', 
            'parametric_forecast (ŷ_pred_t)'
        ]
    
    def compute_shap_values(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        background: Optional[torch.Tensor] = None,
        n_background_samples: int = 10  # Reduced for CPU memory efficiency
    ) -> Dict:
        """
        Compute SHAP values using GradientSHAP (Expected Gradients).
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, 3)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            background: Background distribution for SHAP (n_samples, seq_len, 3).
                       If None, creates baseline from input mean.
            n_background_samples: Number of background samples to use if creating baseline
        
        Returns:
            Dictionary with SHAP values and attributions:
                - shap_values: (seq_len, n_features, horizon) - full attribution
                - feature_importance: (n_features,) - overall importance
                - per_timestep_importance: (n_features, horizon) - per forecast timestep
                - expected_value: Expected model output at baseline
        """
        # Note: We don't set model.eval() here - _compute_gradient_shap handles mode switching
        # because cuDNN RNN backward requires training mode for gradient computation
        device = x.device
        batch_size, seq_len, n_features = x.shape
        
        # Create background distribution if not provided
        if background is None:
            background = self._create_background(x, n_background_samples)
        
        # Ensure background is on the same device
        background = background.to(device)
        
        # Create model wrapper for GradientExplainer
        # Note: wrapper inherits model's current training mode
        wrapper = DELPHIModelWrapper(model, parametric_forecast)
        wrapper.to(device)
        
        # Get horizon from a test forward pass
        with torch.no_grad():
            test_output = wrapper(x[:1])
            horizon = test_output.shape[1]
        
        # Compute SHAP values using GradientExplainer
        shap_values, expected_value = self._compute_gradient_shap(
            wrapper, x, background, horizon
        )
        
        # MEMORY OPTIMIZATION: Delete wrapper immediately (no longer needed)
        del wrapper
        gc.collect()
        
        # shap_values shape: (batch, seq_len, n_features, horizon)
        # Average over batch to get (seq_len, n_features, horizon)
        attributions = shap_values.mean(axis=0)
        
        # MEMORY OPTIMIZATION: Delete large array immediately after processing
        del shap_values
        gc.collect()
        
        # Compute overall feature importance (aggregated over time and horizon)
        feature_importance = np.abs(attributions).sum(axis=(0, 2))  # (n_features,)
        feature_importance = feature_importance / (feature_importance.sum() + 1e-8)
        
        # Compute per-timestep feature importance (aggregated over input sequence)
        per_timestep_importance = np.abs(attributions).sum(axis=0)  # (n_features, horizon)
        per_timestep_importance = per_timestep_importance / (
            per_timestep_importance.sum(axis=0, keepdims=True) + 1e-8
        )
        
        return {
            'shap_values': attributions,  # (seq_len, n_features, horizon)
            'feature_importance': feature_importance,  # (n_features,)
            'per_timestep_importance': per_timestep_importance,  # (n_features, horizon)
            'feature_names': self.feature_names,
            'expected_value': expected_value,  # (horizon,) - baseline prediction
            'attribution_shape': attributions.shape
        }
    
    def _create_background(
        self, 
        x: torch.Tensor, 
        n_samples: int
    ) -> torch.Tensor:
        """
        Create background distribution for GradientSHAP.
        
        Uses mean baseline expanded to n_samples for stable gradient estimation.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            n_samples: Number of background samples
        
        Returns:
            Background tensor (n_samples, seq_len, n_features)
        """
        # Use mean across batch as baseline
        mean_baseline = x.mean(dim=0, keepdim=True)  # (1, seq_len, n_features)
        
        # Add small noise for diversity in background distribution
        # This helps GradientSHAP compute more robust expected gradients
        noise_scale = 0.01 * x.std()
        # Ensure minimum noise scale for numerical stability
        noise_scale = torch.clamp(noise_scale, min=1e-6)
        noise = torch.randn(n_samples, *mean_baseline.shape[1:], device=x.device) * noise_scale
        
        background = mean_baseline.expand(n_samples, -1, -1) + noise
        
        return background
    
    def _compute_gradient_shap(
        self,
        wrapper: nn.Module,
        x: torch.Tensor,
        background: torch.Tensor,
        horizon: int
    ) -> tuple:
        """
        Compute GradientSHAP values for each horizon timestep.
        
        Args:
            wrapper: Model wrapper (nn.Module)
            x: Input tensor (batch, seq_len, n_features)
            background: Background distribution (n_background, seq_len, n_features)
            horizon: Forecast horizon
        
        Returns:
            Tuple of (shap_values, expected_value):
                - shap_values: (batch, seq_len, n_features, horizon)
                - expected_value: (horizon,) - mean prediction on background
        
        Note:
            cuDNN RNN backward pass only works in training mode, so we temporarily
            enable training mode during gradient computation for SHAP values.
        """
        batch_size, seq_len, n_features = x.shape
        device = x.device
        
        # Initialize arrays for SHAP values
        all_shap_values = np.zeros((batch_size, seq_len, n_features, horizon))
        expected_values = np.zeros(horizon)
        
        # Compute expected value (baseline prediction) - can use eval mode for this
        with torch.no_grad():
            baseline_pred = wrapper(background).mean(dim=0)  # (horizon,)
            expected_values = baseline_pred.cpu().numpy()
            del baseline_pred  # Clean up immediately to free memory
            gc.collect()  # Force garbage collection
        
        # IMPORTANT: Enable training mode for cuDNN LSTM backward compatibility
        # cuDNN's optimized RNN backward pass only works in training mode
        # Save the inner model's training state (not just wrapper's) since that's what matters
        inner_model = wrapper.model
        was_training = inner_model.training
        
        # Lock training mode to prevent SHAP from calling eval() internally
        # This is critical because SHAP's GradientExplainer may switch the model to eval mode
        wrapper.lock_training_mode()
        
        try:
            # GradientSHAP for each output dimension (horizon timestep)
            # We need to create separate explainers or handle multi-output
            print(f"        Computing SHAP values for {horizon} horizon timesteps...")
            for t in range(horizon):
                if (t + 1) % 5 == 0 or t == 0 or t == horizon - 1:
                    print(f"          Processing timestep {t+1}/{horizon}...")
                # Create wrapper for single output timestep
                class SingleOutputWrapper(nn.Module):
                    def __init__(self, base_wrapper, timestep):
                        super().__init__()
                        self.base_wrapper = base_wrapper
                        self.timestep = timestep
                    
                    def forward(self, x):
                        full_output = self.base_wrapper(x)
                        return full_output[:, self.timestep:self.timestep+1]
                
                single_wrapper = SingleOutputWrapper(wrapper, t)
                single_wrapper.to(device)
                single_wrapper.train()  # Ensure inner wrapper is also in train mode
                
                # Create GradientExplainer for this timestep
                explainer = shap.GradientExplainer(single_wrapper, background)
                
                # Compute SHAP values
                # Returns list with one element per output (we have 1 output)
                shap_vals = explainer.shap_values(x)
                
                # Handle different SHAP return formats
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]  # Single output
                
                # Convert to numpy
                if isinstance(shap_vals, torch.Tensor):
                    shap_vals = shap_vals.detach().cpu().numpy()
                else:
                    shap_vals = np.array(shap_vals)
                
                # Expected target shape: (batch_size, seq_len, n_features)
                # Some SHAP/model combos return (seq_len, n_features, batch_size) instead
                if shap_vals.shape == (batch_size, seq_len, n_features):
                    # Already in (batch, seq, feature) format
                    pass
                elif shap_vals.shape == (seq_len, n_features, batch_size):
                    # Reorder to (batch, seq, feature)
                    shap_vals = np.transpose(shap_vals, (2, 0, 1))
                else:
                    # Fallback: try to reshape safely if total size matches
                    if shap_vals.size == batch_size * seq_len * n_features:
                        shap_vals = shap_vals.reshape(batch_size, seq_len, n_features)
                    else:
                        raise ValueError(
                            f"Unexpected SHAP values shape {shap_vals.shape}, "
                            f"expected (batch={batch_size}, seq_len={seq_len}, n_features={n_features}) "
                            f"or (seq_len, n_features, batch)."
                        )
                
                # Now guaranteed (batch, seq_len, n_features)
                all_shap_values[:, :, :, t] = shap_vals
                
                # Memory cleanup after each timestep
                del shap_vals, explainer, single_wrapper
                gc.collect()  # Force garbage collection after every timestep
                
                # Additional cleanup for CPU memory
                if device.type == 'cpu':
                    # Clear any cached computations
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Clear CUDA cache if available
            
            print(f"        ✓ Completed SHAP computation for all {horizon} timesteps")
        finally:
            # Unlock training mode and restore original state
            wrapper.unlock_training_mode()
            if not was_training:
                wrapper.eval()  # Recursively sets all children (including inner model) to eval mode
            
            # MEMORY OPTIMIZATION: Delete background tensor after all timesteps complete
            # It's no longer needed after SHAP computation
            del background
            gc.collect()
        
        return all_shap_values, expected_values
    
    def attribute_features(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        background: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Attribute forecast to input features using GradientSHAP.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, 3)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            background: Optional background distribution for SHAP
        
        Returns:
            Dictionary with feature attributions and temporal analysis
        """
        # Compute SHAP attributions
        attribution_results = self.compute_shap_values(
            model, x, parametric_forecast, background, n_background_samples=10  # Reduced for CPU memory efficiency
        )
        
        # Get temporal importance analysis
        temporal_importance = self.temporal_importance(
            attribution_results['shap_values'],
            range(x.shape[1])
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
            shap_values: SHAP values array (seq_len, n_features, horizon)
            timesteps: Range of timesteps
        
        Returns:
            Dictionary with temporal importance analysis:
                - timestep_importance: Importance per input timestep
                - feature_temporal_importance: Feature importance over time
                - top_timesteps: Most important timesteps
        """
        # Handle different shapes
        if len(shap_values.shape) == 3:
            seq_len, n_features, horizon = shap_values.shape
        else:
            # Reshape if needed
            shap_values = shap_values.reshape(-1, shap_values.shape[-2], shap_values.shape[-1])
            seq_len, n_features, horizon = shap_values.shape
        
        # Compute importance per input timestep (sum over features and horizon)
        timestep_importance = np.abs(shap_values).sum(axis=(1, 2))  # (seq_len,)
        
        # Compute importance per feature over time (sum over horizon)
        feature_temporal_importance = np.abs(shap_values).sum(axis=2)  # (seq_len, n_features)
        
        # Find most important timesteps
        n_top = min(5, len(timestep_importance))
        top_timesteps = np.argsort(timestep_importance)[-n_top:][::-1]
        
        return {
            'timestep_importance': timestep_importance.tolist(),
            'feature_temporal_importance': feature_temporal_importance.tolist(),
            'top_timesteps': top_timesteps.tolist(),
            'feature_names': self.feature_names
        }
    
    def verify_additivity(
        self,
        model,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        shap_results: Optional[Dict] = None
    ) -> Dict:
        """
        Verify SHAP additivity property: sum(shap_values) + expected_value ≈ prediction.
        
        Args:
            model: DELPHICore model instance
            x: Input tensor (batch, seq_len, 3)
            parametric_forecast: Parametric baseline forecast
            shap_results: Pre-computed SHAP results (optional)
        
        Returns:
            Dictionary with verification results
        """
        if shap_results is None:
            shap_results = self.compute_shap_values(model, x, parametric_forecast)
        
        # Get actual predictions
        model.eval()
        with torch.no_grad():
            results = model(x, parametric_forecast=parametric_forecast)
            actual_predictions = results['forecast'].cpu().numpy()  # (batch, horizon)
        
        # Compute SHAP-based predictions
        # shap_values: (seq_len, n_features, horizon)
        # expected_value: (horizon,)
        shap_sum = shap_results['shap_values'].sum(axis=(0, 1))  # (horizon,)
        expected_value = shap_results['expected_value']  # (horizon,)
        shap_prediction = shap_sum + expected_value  # (horizon,)
        
        # Compare with mean actual prediction
        mean_actual = actual_predictions.mean(axis=0)  # (horizon,)
        
        # Compute error
        absolute_error = np.abs(shap_prediction - mean_actual)
        relative_error = absolute_error / (np.abs(mean_actual) + 1e-8)
        
        return {
            'shap_prediction': shap_prediction.tolist(),
            'actual_prediction': mean_actual.tolist(),
            'absolute_error': absolute_error.tolist(),
            'relative_error': relative_error.tolist(),
            'mean_absolute_error': float(absolute_error.mean()),
            'mean_relative_error': float(relative_error.mean()),
            'additivity_holds': bool(relative_error.mean() < 0.1)  # Within 10%
        }
