"""
Core DELPHI Model: Integrates Variational HMM Gating with Deep Ensemble Correctors.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import numpy as np

from .hmm_gating import VariationalHMMGating
from .ensemble_correctors import DeepEnsembleCorrectors


class DELPHICore(nn.Module):
    """
    Core DELPHI model combining:
    - Variational HMM gating for regime detection
    - Deep ensemble of specialized RNN correctors
    - Probabilistic uncertainty quantification
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # [z_t, w_t, ŷ_pred_t]
        n_states: int = 4,
        hmm_hidden_size: int = 64,
        hmm_num_layers: int = 2,
        ensemble_hidden_size: int = 64,
        ensemble_num_layers: int = 2,
        output_dim: int = 26,  # Forecast horizon
        n_ensemble_members: int = 5,
        dropout: float = 0.2,
        use_xlstm_for_trend: bool = False
    ):
        """
        Initialize DELPHI core model.
        
        Args:
            input_dim: Input feature dimension
            n_states: Number of HMM latent states
            hmm_hidden_size: HMM LSTM hidden size
            hmm_num_layers: HMM LSTM layers
            ensemble_hidden_size: Ensemble LSTM hidden size
            ensemble_num_layers: Ensemble LSTM layers
            output_dim: Forecast horizon
            n_ensemble_members: Number of ensemble members (M=5)
            dropout: Dropout rate
            use_xlstm_for_trend: Whether to use xLSTMTime for trend corrector
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_states = n_states
        self.output_dim = output_dim
        
        # Variational HMM Gating
        self.hmm_gating = VariationalHMMGating(
            input_dim=input_dim,
            n_states=n_states,
            hidden_size=hmm_hidden_size,
            num_layers=hmm_num_layers,
            dropout=dropout
        )
        
        # Deep Ensemble Correctors
        self.ensemble = DeepEnsembleCorrectors(
            input_dim=input_dim,
            hidden_size=ensemble_hidden_size,
            num_layers=ensemble_num_layers,
            output_dim=output_dim,
            dropout=dropout,
            n_members=n_ensemble_members,
            use_xlstm_for_trend=use_xlstm_for_trend
        )
    
    def forward(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DELPHI model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            return_states: Whether to return HMM states
        
        Returns:
            Dictionary with:
                - 'correction': Correction from ensemble (batch, output_dim)
                - 'forecast': Final forecast (batch, output_dim)
                - 'states': HMM states if return_states=True
                - 'state_probs': HMM state probabilities
        """
        batch_size = x.shape[0]
        
        # Get HMM state probabilities (posterior)
        state_probs = self.hmm_gating(x, mode='posterior')
        
        # Sample states for routing
        states = torch.multinomial(state_probs, 1).squeeze(-1)  # (batch,)
        states_expanded = states.unsqueeze(1).expand(-1, x.shape[1])  # (batch, seq_len)
        
        # Get correction from ensemble with state-based routing
        correction = self.ensemble(x, states=states_expanded)
        
        # Combine with parametric forecast
        if parametric_forecast is not None:
            forecast = parametric_forecast + correction
        else:
            # If no parametric forecast, use correction only
            forecast = correction
        
        result = {
            'correction': correction,
            'forecast': forecast,
            'state_probs': state_probs
        }
        
        if return_states:
            result['states'] = states
        
        return result
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        parametric_forecast: Optional[torch.Tensor] = None,
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty quantification via HMM trajectory sampling.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            parametric_forecast: Parametric baseline forecast (batch, output_dim)
            num_samples: Number of HMM trajectory samples
        
        Returns:
            Dictionary with:
                - 'mean': Mean forecast (batch, output_dim)
                - 'std': Standard deviation (batch, output_dim)
                - 'forecasts': All sampled forecasts (num_samples, batch, output_dim)
                - 'confidence_intervals': 95% CI (batch, output_dim, 2)
        """
        # Sample HMM trajectories
        trajectories = self.hmm_gating.sample_trajectories(x, num_samples)
        # trajectories: (num_samples, batch, seq_len)
        
        # Get ensemble uncertainty
        mean_correction, std_correction = self.ensemble.forward_with_uncertainty(x, num_samples)
        
        # Sample corrections for each trajectory
        forecasts = []
        for i in range(num_samples):
            traj = trajectories[i]  # (batch, seq_len)
            # Use trajectory to route ensemble
            correction = self.ensemble(x, states=traj)
            
            if parametric_forecast is not None:
                forecast = parametric_forecast + correction
            else:
                forecast = correction
            
            forecasts.append(forecast)
        
        forecasts = torch.stack(forecasts, dim=0)  # (num_samples, batch, output_dim)
        
        # Compute statistics
        mean_forecast = forecasts.mean(dim=0)
        std_forecast = forecasts.std(dim=0)
        
        # 95% confidence intervals
        lower = mean_forecast - 1.96 * std_forecast
        upper = mean_forecast + 1.96 * std_forecast
        confidence_intervals = torch.stack([lower, upper], dim=-1)  # (batch, output_dim, 2)
        
        return {
            'mean': mean_forecast,
            'std': std_forecast,
            'forecasts': forecasts,
            'confidence_intervals': confidence_intervals
        }
    
    def get_regime_explanation(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get interpretable regime explanation from HMM states.
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary with regime information
        """
        state_probs = self.hmm_gating.get_state_probs(x)
        transition_matrix = self.hmm_gating.get_transition_matrix(x)
        
        return {
            'state_probs': state_probs,
            'transition_matrix': transition_matrix,
            'dominant_state': torch.argmax(state_probs, dim=-1)
        }


