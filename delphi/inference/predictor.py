"""
Inference utilities for DELPHI: Prediction with uncertainty quantification.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from ..models.delphi_core import DELPHICore
from ..data.preprocessing import prepare_model_inputs


class DELPHIPredictor:
    """
    Predictor for DELPHI model with uncertainty quantification.
    """
    
    def __init__(
        self,
        model: DELPHICore,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize DELPHI predictor.
        
        Args:
            model: Trained DELPHI model
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_config: Model configuration dictionary
            device: Device for inference
        """
        # Force CPU if CUDA not available (handles models trained on GPU)
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        
        model = DELPHICore(**model_config)
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=torch.device(device),
            weights_only=False  # Required for checkpoints with numpy arrays
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return cls(model, device)
    
    def predict(
        self,
        inputs: np.ndarray,
        parametric_forecasts: Optional[np.ndarray] = None,
        num_samples: int = 100,
        return_uncertainty: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty.
        
        Args:
            inputs: Input sequences (n_samples, seq_len, input_dim)
            parametric_forecasts: Parametric baseline forecasts (n_samples, forecast_horizon)
            num_samples: Number of HMM trajectory samples
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Convert to tensors
        inputs_tensor = torch.FloatTensor(inputs).to(self.device)
        if parametric_forecasts is not None:
            param_tensor = torch.FloatTensor(parametric_forecasts).to(self.device)
        else:
            param_tensor = None
        
        with torch.no_grad():
            if return_uncertainty:
                results = self.model.predict_with_uncertainty(
                    inputs_tensor,
                    parametric_forecast=param_tensor,
                    num_samples=num_samples
                )
                
                return {
                    'mean': results['mean'].cpu().numpy(),
                    'std': results['std'].cpu().numpy(),
                    'forecasts': results['forecasts'].cpu().numpy(),
                    'confidence_intervals': results['confidence_intervals'].cpu().numpy()
                }
            else:
                results = self.model(inputs_tensor, parametric_forecast=param_tensor)
                return {
                    'forecast': results['forecast'].cpu().numpy(),
                    'correction': results['correction'].cpu().numpy(),
                    'state_probs': results['state_probs'].cpu().numpy()
                }
    
    def predict_series(
        self,
        main_signal: np.ndarray,
        weak_signal_ratio: Optional[np.ndarray] = None,
        parametric_forecast: Optional[np.ndarray] = None,
        num_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Predict for a single time series.
        
        Args:
            main_signal: Main time series signal
            weak_signal_ratio: Weak signal ratio
            parametric_forecast: Parametric baseline forecast
            num_samples: Number of samples for uncertainty
        
        Returns:
            Prediction results
        """
        # Prepare input
        input_tensor = prepare_model_inputs(
            main_signal,
            weak_signal_ratio=weak_signal_ratio,
            parametric_forecast=parametric_forecast
        )
        
        # Add batch dimension
        input_tensor = input_tensor.reshape(1, *input_tensor.shape)
        
        if parametric_forecast is not None:
            param_tensor = parametric_forecast.reshape(1, -1)
        else:
            param_tensor = None
        
        return self.predict(
            input_tensor,
            parametric_forecasts=param_tensor,
            num_samples=num_samples
        )
    
    def get_regime_explanation(
        self,
        inputs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get interpretable regime explanation.
        
        Args:
            inputs: Input sequences
        
        Returns:
            Regime information
        """
        inputs_tensor = torch.FloatTensor(inputs).to(self.device)
        
        with torch.no_grad():
            explanation = self.model.get_regime_explanation(inputs_tensor)
            
            return {
                'state_probs': explanation['state_probs'].cpu().numpy(),
                'transition_matrix': explanation['transition_matrix'].cpu().numpy(),
                'dominant_state': explanation['dominant_state'].cpu().numpy()
            }


